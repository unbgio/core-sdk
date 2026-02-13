fn main() {
    uniffi::generate_scaffolding("./src/unbg.udl").expect("failed to generate uniffi scaffolding");
}
