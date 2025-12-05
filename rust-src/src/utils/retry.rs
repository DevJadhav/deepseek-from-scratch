use std::time::Duration;
use tokio::time::sleep;
use crate::utils::error::{DeepSeekError, Result};

pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_factor: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_factor: 2.0,
        }
    }
}

pub async fn retry_with_backoff<F, Fut, T>(
    operation: F,
    policy: &RetryPolicy,
) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut current_delay = policy.initial_delay;
    let mut attempts = 0;

    loop {
        attempts += 1;
        match operation().await {
            Ok(value) => return Ok(value),
            Err(e) => {
                if attempts > policy.max_retries {
                    return Err(e);
                }

                // Log the retry attempt (assuming we have logging set up, otherwise println for now)
                // In a real app, use the logging module.
                eprintln!("Operation failed (attempt {}/{}): {}. Retrying in {:?}...", 
                    attempts, policy.max_retries, e, current_delay);

                sleep(current_delay).await;

                current_delay = (current_delay.mul_f64(policy.backoff_factor)).min(policy.max_delay);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[tokio::test]
    async fn test_retry_success() {
        let policy = RetryPolicy::default();
        let result = retry_with_backoff(|| async { Ok::<_, DeepSeekError>(42) }, &policy).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_retry_eventual_success() {
        let policy = RetryPolicy {
            max_retries: 3,
            initial_delay: Duration::from_millis(1),
            ..Default::default()
        };
        
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = counter.clone();

        let result = retry_with_backoff(move || {
            let counter = counter_clone.clone();
            async move {
                let mut num = counter.lock().unwrap();
                *num += 1;
                if *num < 3 {
                    Err(DeepSeekError::Training("Temporary failure".to_string()))
                } else {
                    Ok("Success")
                }
            }
        }, &policy).await;

        assert_eq!(result.unwrap(), "Success");
        assert_eq!(*counter.lock().unwrap(), 3);
    }

    #[tokio::test]
    async fn test_retry_failure() {
        let policy = RetryPolicy {
            max_retries: 2,
            initial_delay: Duration::from_millis(1),
            ..Default::default()
        };

        let result = retry_with_backoff(|| async {
            Err::<(), _>(DeepSeekError::Training("Persistent failure".to_string()))
        }, &policy).await;

        assert!(result.is_err());
    }
}
