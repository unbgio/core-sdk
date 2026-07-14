const COMMANDS: &[&str] = &["tauri_remove_background_command"];

fn main() {
    // Registers plugin metadata (including ACL permission discovery) for consumers.
    tauri_plugin::Builder::new(COMMANDS).build();
}
