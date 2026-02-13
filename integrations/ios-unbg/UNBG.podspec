Pod::Spec.new do |s|
  s.name             = 'UNBG'
  s.version          = '0.1.0'
  s.summary          = 'UNBG iOS background removal SDK'
  s.description      = 'XCFramework packaging for UNBG Rust/UniFFI runtime.'
  s.homepage         = 'https://example.com/unbg'
  s.license          = { :type => 'MIT' }
  s.author           = { 'UNBG' => 'team@example.com' }
  s.platform         = :ios, '13.0'
  s.vendored_frameworks = 'dist/UNBG.xcframework'
  s.source           = { :path => '.' }
  s.swift_version    = '5.9'
end
