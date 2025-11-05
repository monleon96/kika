// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::Manager;
use std::env;

// Backend URL configuration
// Defaults to production but can be overridden with KIKA_BACKEND_URL env var
fn get_backend_url() -> String {
    env::var("KIKA_BACKEND_URL")
        .unwrap_or_else(|_| "https://kika-backend.onrender.com".to_string())
}

// Test backend connection
#[tauri::command]
async fn check_backend_health() -> Result<bool, String> {
    let backend_url = get_backend_url();
    let url = format!("{}/healthz", backend_url);
    
    match reqwest::get(&url).await {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

// Call Python API endpoint
#[tauri::command]
async fn call_backend_api(
    endpoint: String,
    method: String,
    body: Option<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let backend_url = get_backend_url();
    let url = format!("{}{}", backend_url, endpoint);
    let client = reqwest::Client::new();
    
    let response = match method.as_str() {
        "GET" => client.get(&url).send().await,
        "POST" => {
            let req = client.post(&url);
            if let Some(data) = body {
                req.json(&data).send().await
            } else {
                req.send().await
            }
        }
        _ => return Err("Unsupported HTTP method".to_string()),
    };
    
    match response {
        Ok(resp) => {
            match resp.json::<serde_json::Value>().await {
                Ok(data) => Ok(data),
                Err(e) => Err(format!("Failed to parse response: {}", e)),
            }
        }
        Err(e) => Err(format!("Request failed: {}", e)),
    }
}

// Read file from local filesystem
#[tauri::command]
async fn read_local_file(path: String) -> Result<String, String> {
    match std::fs::read_to_string(&path) {
        Ok(content) => Ok(content),
        Err(e) => Err(format!("Failed to read file: {}", e)),
    }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            check_backend_health,
            call_backend_api,
            read_local_file,
        ])
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                let window = app.get_window("main").unwrap();
                window.open_devtools();
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
