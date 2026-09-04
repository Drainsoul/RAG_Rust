mod clean;
mod hash;
mod database;
use database::Database;

use reqwest;
use anyhow::Result;
use std::sync::Arc;
use serde_json::{Value, json};
use actix_web::{
    web, App, HttpServer, HttpResponse,
    dev::Service,
    http::header,
};

struct RAGState {
    database: Database,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let db = Database::new();

    let documents: Vec<String> = load_data("data/text.txt").await.unwrap_or_default();
    for document in documents {
        let _ = db.add_document(&document).await;
    }

    let state = Arc::new(RAGState { database: db });

    let host = "0.0.0.0:3000";
    println!("Starting Server http://{host}");

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(state.clone()))
            .wrap_fn(|mut req, srv| {
                req.headers_mut().insert(
                    header::CONTENT_TYPE,
                    header::HeaderValue::from_static("application/json"),
                );
                srv.call(req)
            })
            .route("/", web::post().to(root))
            .route("/doc", web::post().to(doc))
    })
    .bind(host)?
    .run()
    .await
}

async fn ask(question: &str) -> Result<String> {
    let client = reqwest::Client::new();
    let response: String = client
        .get(&format!("http://0.0.0.0:8000/{question}"))
        .send()
        .await?
        .text()
        .await?;

    Ok(response)
}

async fn load_data(filename: &str) -> Result<Vec<String>> {
    let contents = tokio::fs::read_to_string(filename).await?;
    let lines: Vec<String> = contents
        .lines()
        .map(|line| line.to_string())
        .collect();

    Ok(lines)
}

async fn root(
    state: web::Data<Arc<RAGState>>,
    body: web::Json<Value>,
) -> HttpResponse {
    let prompt: &str = body["prompt"].as_str().unwrap_or("");
    let docs = state.database.search(prompt, 2).await.unwrap_or("".to_string());
    let answer = ask(&format!("{docs}\n{prompt}")).await.unwrap_or("".to_string());

    println!("{prompt}");
    println!("{answer}");

    HttpResponse::Ok().json(json!({"answer": answer}))
}

async fn doc(
    state: web::Data<Arc<RAGState>>,
    body: web::Json<Value>,
) -> HttpResponse {
    let document: &str = body["document"].as_str().unwrap_or("");
    let _ = state.database.add_document(document).await;

    HttpResponse::Ok().json(json!({"text": "Doc loaded successfully!"}))
}