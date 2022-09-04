use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader};
use std::path::Path;

use anyhow::Result;

// Appropriated: https://github.com/philschmid/rust-machine-learning/blob/master/pipeline-rs/src/modeling_utils.rs

#[derive(Debug, Deserialize)]
pub struct Config {
    pub id2label: HashMap<usize, String>,
}

impl Config {
    pub fn from_file(config_path: &Path) -> Result<Self> {
        // Open the file in read-only mode with buffer.
        let file = File::open(config_path)?;
        let reader = BufReader::new(file);

        // Read the JSON contents of the file as an instance of `User`.
        let u: Config = serde_json::from_reader(reader)?;
        Ok(u)
    }
}
