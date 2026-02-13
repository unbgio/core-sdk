#[derive(Debug, Clone)]
pub struct BenchmarkCase {
    pub name: String,
    pub width: u32,
    pub height: u32,
}

pub fn default_cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase {
            name: "small".to_string(),
            width: 512,
            height: 512,
        },
        BenchmarkCase {
            name: "medium".to_string(),
            width: 1024,
            height: 1024,
        },
        BenchmarkCase {
            name: "large".to_string(),
            width: 2048,
            height: 2048,
        },
    ]
}

pub fn describe(cases: &[BenchmarkCase]) -> String {
    cases
        .iter()
        .map(|c| format!("{}:{}x{}", c.name, c.width, c.height))
        .collect::<Vec<_>>()
        .join(", ")
}
