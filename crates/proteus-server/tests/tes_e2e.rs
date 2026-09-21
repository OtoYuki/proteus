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
            path: "inputs/candidate.fasta".into(),
            type_: TesFileType::File,
            content: Some(fasta_content.into()),
        }],
        executors: vec![TesExecutor {
            image: "".into(),
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
            workdir: None,
            stdout: Some("outputs/stdout.log".into()),
            stderr: Some("outputs/stderr.log".into()),
            stdin: None,
            env: std::collections::HashMap::new(),
            ignore_error: false,
        }],
        outputs: vec![
            TesOutput {
                name: Some("predicted_pdb".into()),
                description: None,
                url: None,
                path: "outputs/structure.pdb".into(),
                type_: TesFileType::File,
            },
            TesOutput {
                name: Some("screening_report".into()),
                description: None,
                url: None,
                path: "outputs/report.txt".into(),
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
