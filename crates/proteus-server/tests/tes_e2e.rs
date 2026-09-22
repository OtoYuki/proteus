use proteus_core::tes::{TesExecutor, TesFileType, TesInput, TesOutput, TesState, TesTask};
use proteus_engine::simulated::SimulatedRunner;
use proteus_engine::PipelineScheduler;
use proteus_server::{build_router, AppState};
use proteus_storage::pool::create_in_memory_pool;
use proteus_storage::repository::ProteusRepository;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;
use tokio::net::TcpListener;

#[tokio::test]
async fn test_tes_v1_nextflow_e2e_lifecycle() {
    // 1. Initialize in-memory storage, CAS, and scheduler
    let pool = create_in_memory_pool().await.unwrap();
    let repo = ProteusRepository::new(pool);
    let runner = Arc::new(SimulatedRunner::new());
    let tmp = tempdir().unwrap();
    let artifacts_dir = tmp.path().to_path_buf();

    let scheduler = PipelineScheduler::new(repo, runner, artifacts_dir);
    let state = AppState::new(scheduler);
    let router = build_router(state);

    // 2. Bind TCP listener on random available OS port
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base_url = format!("http://{}", addr);

    // Spawn server in background tokio task
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let client = reqwest::Client::new();

    // 3. Query Service Info (Service Discovery)
    let resp = client
        .get(format!("{}/v1/service-info", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let s_info: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(s_info["id"], "org.ga4gh.proteus");
    assert_eq!(s_info["type"]["artifact"], "tes");
    assert_eq!(s_info["type"]["version"], "1.1.0");

    // 4. Submit Nextflow-style TES task
    let fasta_content = ">test_candidate\nTTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN\n";
    let task_payload = TesTask {
        name: Some("nextflow_nf_ga4gh_screening_step".into()),
        description: Some("Simulated Nextflow TES bio-compute task".into()),
        inputs: vec![TesInput {
            name: Some("variant_fasta".into()),
            description: None,
            url: None,
            path: "/work/inputs/candidate.fasta".into(),
            type_: TesFileType::File,
            content: Some(fasta_content.into()),
        }],
        executors: vec![TesExecutor {
            image: "docker.io/library/alpine:3.20".into(),
            command: vec![
                "sh".into(),
                "-c".into(),
                r#"
                echo "Folding sequence from input..."
                mkdir -p outputs
                cat << 'EOF' > outputs/structure.pdb
HEADER    PREDICTED STRUCTURE
ATOM      1  N   THR A   1      17.047  14.099   3.625  1.00 13.79           N
ATOM      2  CA  THR A   1      16.967  12.784   4.338  1.00 10.80           C
ATOM      3  C   THR A   1      15.685  12.755   5.133  1.00  9.19           C
ATOM      4  O   THR A   1      15.268  13.825   5.594  1.00  9.85           O
TER       5      THR A   1
END
EOF
                echo "Folded structure generated successfully." > outputs/report.txt
                "#
                .into(),
            ],
            workdir: Some("/work".into()),
            stdout: Some("/work/outputs/stdout.log".into()),
            stderr: Some("/work/outputs/stderr.log".into()),
            stdin: None,
            env: std::collections::HashMap::new(),
            ignore_error: false,
        }],
        outputs: vec![
            TesOutput {
                name: Some("predicted_pdb".into()),
                description: None,
                url: None,
                path: "/work/outputs/structure.pdb".into(),
                type_: TesFileType::File,
            },
            TesOutput {
                name: Some("screening_report".into()),
                description: None,
                url: None,
                path: "/work/outputs/report.txt".into(),
                type_: TesFileType::File,
            },
        ],
        ..Default::default()
    };

    let resp = client
        .post(format!("{}/v1/tasks", base_url))
        .json(&task_payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let create_body: serde_json::Value = resp.json().await.unwrap();
    let task_id = create_body["id"].as_str().unwrap().to_string();
    assert!(!task_id.is_empty());

    // 5. Poll until COMPLETE (as Nextflow does)
    let mut completed = false;
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let poll_resp = client
            .get(format!("{}/v1/tasks/{}?view=MINIMAL", base_url, task_id))
            .send()
            .await
            .unwrap();
        if poll_resp.status() == reqwest::StatusCode::OK {
            let task_state: serde_json::Value = poll_resp.json().await.unwrap();
            let state_str = task_state["state"].as_str().unwrap_or("");
            if state_str == "COMPLETE" {
                completed = true;
                break;
            }
        }
    }
    assert!(completed, "TES task did not reach COMPLETE state in time");

    // 6. Query FULL view and verify outputs, logs, and biophysics
    let resp = client
        .get(format!("{}/v1/tasks/{}?view=FULL", base_url, task_id))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let full_task: TesTask = resp.json().await.unwrap();
    assert_eq!(full_task.state, TesState::Complete);
    assert_eq!(full_task.logs.len(), 1);
    assert_eq!(full_task.logs[0].logs.len(), 1);
    assert_eq!(full_task.logs[0].logs[0].exit_code, Some(0));
    assert!(full_task.logs[0].logs[0]
        .stdout
        .as_ref()
        .unwrap()
        .contains("Folding sequence"));
    assert_eq!(full_task.logs[0].outputs.len(), 2);

    // 7. Verify Prometheus Observability
    let metrics_resp = client
        .get(format!("{}/metrics", base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(metrics_resp.status(), reqwest::StatusCode::OK);
    let metrics_text = metrics_resp.text().await.unwrap();
    assert!(metrics_text.contains("proteus_tasks_total{status=\"complete\"} 1"));
    assert!(metrics_text.contains("proteus_cas_operations_total"));
    assert!(metrics_text.contains("proteus_http_requests_total"));
}

/// Spin up a router-backed server with the given options; returns the base URL.
async fn spawn(
    options: proteus_server::ServerOptions,
    tes: Option<proteus_engine::TesExecutionConfig>,
) -> String {
    let pool = create_in_memory_pool().await.unwrap();
    let repo = ProteusRepository::new(pool);
    let runner = Arc::new(SimulatedRunner::new());
    let tmp = tempdir().unwrap();
    let mut scheduler = PipelineScheduler::new(repo, runner, tmp.path().to_path_buf());
    if let Some(tes) = tes {
        scheduler = scheduler.with_tes_config(tes);
    }
    let router = proteus_server::build_router_with_options(AppState::new(scheduler), options);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    std::mem::forget(tmp);
    format!("http://{addr}")
}

fn minimal_task(image: &str) -> TesTask {
    TesTask {
        executors: vec![TesExecutor {
            image: image.into(),
            command: vec!["true".into()],
            workdir: Some("/work".into()),
            stdout: None,
            stderr: None,
            stdin: None,
            env: std::collections::HashMap::new(),
            ignore_error: false,
        }],
        ..Default::default()
    }
}

#[tokio::test]
async fn bearer_token_guards_tes_and_native_apis_but_not_health() {
    let base = spawn(
        proteus_server::ServerOptions {
            auth_token: Some("s3cret".into()),
        },
        None,
    )
    .await;
    let client = reqwest::Client::new();
    assert_eq!(
        client
            .get(format!("{base}/health"))
            .send()
            .await
            .unwrap()
            .status(),
        reqwest::StatusCode::OK
    );
    assert_eq!(
        client
            .get(format!("{base}/metrics"))
            .send()
            .await
            .unwrap()
            .status(),
        reqwest::StatusCode::OK
    );
    let r = client
        .get(format!("{base}/v1/service-info"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::UNAUTHORIZED);
    assert_eq!(r.headers().get("www-authenticate").unwrap(), "Bearer");
    let r = client
        .get(format!("{base}/v1/service-info"))
        .bearer_auth("wrong")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::UNAUTHORIZED);
    let r = client
        .get(format!("{base}/v1/service-info"))
        .bearer_auth("s3cret")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::OK);
    let r = client
        .post(format!("{base}/api/v1/sequences"))
        .json(&serde_json::json!({"header": "x", "fasta": ">x\nACD\n"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn image_allowlist_rejects_with_400_and_is_advertised() {
    let tes = proteus_engine::TesExecutionConfig {
        allow_images: vec![glob::Pattern::new("ghcr.io/otoyuki/*").unwrap()],
        ..Default::default()
    };
    let base = spawn(Default::default(), Some(tes)).await;
    let client = reqwest::Client::new();
    let info: serde_json::Value = client
        .get(format!("{base}/v1/service-info"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(info["tags"]["proteus.executor"], "host");
    assert_eq!(info["tags"]["proteus.image_allowlist"], "ghcr.io/otoyuki/*");

    let r = client
        .post(format!("{base}/v1/tasks"))
        .json(&minimal_task("docker.io/library/alpine:3.20"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: serde_json::Value = r.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("not allowed"));

    let r = client
        .post(format!("{base}/v1/tasks"))
        .json(&minimal_task("ghcr.io/otoyuki/proteus:0.3.0"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::OK);
}

#[tokio::test]
async fn relative_or_system_paths_are_rejected() {
    let base = spawn(Default::default(), None).await;
    let client = reqwest::Client::new();
    let mut t = minimal_task("docker.io/library/alpine:3.20");
    t.outputs.push(TesOutput {
        name: None,
        description: None,
        url: None,
        path: "relative/out.txt".into(),
        type_: TesFileType::File,
    });
    let r = client
        .post(format!("{base}/v1/tasks"))
        .json(&t)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::BAD_REQUEST);
    let mut t = minimal_task("docker.io/library/alpine:3.20");
    t.executors[0].workdir = Some("/etc".into());
    let r = client
        .post(format!("{base}/v1/tasks"))
        .json(&t)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn file_url_allowlist_rejects_with_400_and_is_advertised() {
    let dir = tempdir().unwrap();
    let tes = proteus_engine::TesExecutionConfig {
        allow_dirs: vec![dir.path().to_path_buf()],
        ..Default::default()
    };
    let base = spawn(Default::default(), Some(tes)).await;
    let client = reqwest::Client::new();
    let info: serde_json::Value = client
        .get(format!("{base}/v1/service-info"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        info["tags"]["proteus.file_allowlist"],
        dir.path().display().to_string()
    );

    let mut t = minimal_task("docker.io/library/alpine:3.20");
    t.outputs.push(TesOutput {
        name: None,
        description: None,
        url: Some("file:///etc/leak.txt".into()),
        path: "/work/out.txt".into(),
        type_: TesFileType::File,
    });
    let r = client
        .post(format!("{base}/v1/tasks"))
        .json(&t)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: serde_json::Value = r.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("allows"));
}

#[tokio::test]
async fn list_tasks_pages_and_filters_in_sql() {
    let base = spawn(proteus_server::ServerOptions::default(), None).await;
    let client = reqwest::Client::new();

    // Five tasks: alpha-1..3 (tagged run=a) and beta-1..2 (tagged run=b).
    for (name, run) in [
        ("alpha-1", "a"),
        ("alpha-2", "a"),
        ("alpha-3", "a"),
        ("beta-1", "b"),
        ("beta-2", "b"),
    ] {
        let mut task = minimal_task("docker.io/library/alpine:3.20");
        task.name = Some(name.into());
        task.tags.insert("run".into(), run.into());
        let resp = client
            .post(format!("{base}/v1/tasks"))
            .json(&task)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), reqwest::StatusCode::OK);
    }

    let get = |url: String| {
        let client = client.clone();
        async move {
            let resp = client.get(url).send().await.unwrap();
            assert_eq!(resp.status(), reqwest::StatusCode::OK);
            resp.json::<serde_json::Value>().await.unwrap()
        }
    };

    // Page through everything two at a time: every task exactly once, no duplicates.
    let mut seen = Vec::new();
    let mut token: Option<String> = None;
    loop {
        let mut url = format!("{base}/v1/tasks?page_size=2&view=BASIC");
        if let Some(t) = &token {
            url.push_str(&format!("&page_token={t}"));
        }
        let page = get(url).await;
        let tasks = page["tasks"].as_array().unwrap();
        assert!(tasks.len() <= 2);
        seen.extend(tasks.iter().map(|t| t["id"].as_str().unwrap().to_string()));
        match page["next_page_token"].as_str() {
            Some(t) => token = Some(t.to_string()),
            None => break,
        }
    }
    assert_eq!(
        seen.len(),
        5,
        "paged listing must cover every task: {seen:?}"
    );
    let mut dedup = seen.clone();
    dedup.sort();
    dedup.dedup();
    assert_eq!(dedup.len(), 5, "duplicates across pages: {seen:?}");

    // name_prefix is applied in SQL.
    let page = get(format!("{base}/v1/tasks?name_prefix=alpha&view=BASIC")).await;
    let names: Vec<&str> = page["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(names.len(), 3, "{names:?}");
    assert!(names.iter().all(|n| n.starts_with("alpha")), "{names:?}");
    assert!(page["next_page_token"].is_null());

    // `%` and `_` in the prefix are literals, not LIKE wildcards.
    let page = get(format!("{base}/v1/tasks?name_prefix=%25&view=BASIC")).await;
    assert_eq!(page["tasks"].as_array().unwrap().len(), 0);
    let page = get(format!("{base}/v1/tasks?name_prefix=alpha_&view=BASIC")).await;
    assert_eq!(page["tasks"].as_array().unwrap().len(), 0);

    // Tag filters page over the filtered set.
    let page = get(format!(
        "{base}/v1/tasks?tag_key=run&tag_value=b&page_size=1&view=BASIC"
    ))
    .await;
    assert_eq!(page["tasks"].as_array().unwrap().len(), 1);
    assert!(page["tasks"][0]["name"]
        .as_str()
        .unwrap()
        .starts_with("beta"));
    let token = page["next_page_token"].as_str().unwrap().to_string();
    let page = get(format!(
        "{base}/v1/tasks?tag_key=run&tag_value=b&page_size=1&page_token={token}&view=BASIC"
    ))
    .await;
    assert_eq!(page["tasks"].as_array().unwrap().len(), 1);
    assert!(page["tasks"][0]["name"]
        .as_str()
        .unwrap()
        .starts_with("beta"));
    assert!(page["next_page_token"].is_null());

    // A state filter that matches nothing yields an empty page without a token.
    let page = get(format!("{base}/v1/tasks?state=CANCELED&view=BASIC")).await;
    assert_eq!(page["tasks"].as_array().unwrap().len(), 0);
    assert!(page["next_page_token"].is_null());
}
