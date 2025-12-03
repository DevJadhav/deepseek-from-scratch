use super::CollectiveCommunicator;
use candle_core::{Result, Tensor, Device};
use std::sync::{Arc, Mutex, Barrier};

struct SharedState {
    buffers: Mutex<Vec<Option<Tensor>>>,
    barrier: Barrier,
}

pub struct LocalCommunicator {
    rank: usize,
    world_size: usize,
    shared: Arc<SharedState>,
}

impl LocalCommunicator {
    pub fn new_group(world_size: usize) -> Vec<Self> {
        let shared = Arc::new(SharedState {
            buffers: Mutex::new((0..world_size).map(|_| None).collect()),
            barrier: Barrier::new(world_size),
        });

        (0..world_size)
            .map(|rank| Self {
                rank,
                world_size,
                shared: shared.clone(),
            })
            .collect()
    }
}

impl CollectiveCommunicator for LocalCommunicator {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn all_reduce(&self, tensor: &Tensor) -> Result<Tensor> {
        // 1. Write to buffer
        {
            let mut buffers = self.shared.buffers.lock().unwrap();
            buffers[self.rank] = Some(tensor.clone());
        }
        
        // 2. Wait for all
        self.shared.barrier.wait();
        
        // 3. Compute sum (only need to do this once effectively, or everyone does it)
        // For simplicity, everyone computes it.
        let sum = {
            let buffers = self.shared.buffers.lock().unwrap();
            let mut sum = buffers[0].as_ref().unwrap().clone();
            for i in 1..self.world_size {
                sum = (sum + buffers[i].as_ref().unwrap())?;
            }
            sum
        };
        
        // 4. Wait for all to read before clearing (optional if we overwrite next time, but safer)
        self.shared.barrier.wait();
        
        Ok(sum)
    }

    fn all_gather(&self, tensor: &Tensor) -> Result<Tensor> {
        // 1. Write to buffer
        {
            let mut buffers = self.shared.buffers.lock().unwrap();
            buffers[self.rank] = Some(tensor.clone());
        }
        
        // 2. Wait for all
        self.shared.barrier.wait();
        
        // 3. Gather
        let gathered = {
            let buffers = self.shared.buffers.lock().unwrap();
            let tensors: Vec<&Tensor> = buffers.iter().map(|t| t.as_ref().unwrap()).collect();
            Tensor::cat(&tensors, 0)?
        };
        
        self.shared.barrier.wait();
        
        Ok(gathered)
    }

    fn broadcast(&self, tensor: &Tensor, root_rank: usize) -> Result<Tensor> {
        // 1. Root writes
        if self.rank == root_rank {
            let mut buffers = self.shared.buffers.lock().unwrap();
            buffers[root_rank] = Some(tensor.clone());
        }
        
        // 2. Wait
        self.shared.barrier.wait();
        
        // 3. Read
        let result = {
            let buffers = self.shared.buffers.lock().unwrap();
            buffers[root_rank].as_ref().unwrap().clone()
        };
        
        self.shared.barrier.wait();
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_all_reduce() -> Result<()> {
        let world_size = 4;
        let comms = LocalCommunicator::new_group(world_size);
        
        let handles: Vec<_> = comms.into_iter().map(|comm| {
            thread::spawn(move || {
                let device = Device::Cpu;
                let t = Tensor::new(&[1.0f32], &device).unwrap();
                let res = comm.all_reduce(&t).unwrap();
                // Get the first (and only) element
                res.get(0).unwrap().to_scalar::<f32>().unwrap()
            })
        }).collect();

        for h in handles {
            let val = h.join().unwrap();
            assert_eq!(val, 4.0);
        }
        Ok(())
    }
}
