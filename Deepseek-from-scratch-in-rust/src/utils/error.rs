use thiserror::Error;

#[derive(Error, Debug)]
pub enum DeepSeekError {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model configuration error: {0}")]
    Config(String),

    #[error("Training error: {0}")]
    Training(String),
    
    #[error("Not implemented: {0}")]
    NotImplemented(String),
    
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),
    
    #[error("Distributed error: {0}")]
    Distributed(String),
    
    #[error("Communication error: {0}")]
    Communication(String),
}

pub type Result<T> = std::result::Result<T, DeepSeekError>;
