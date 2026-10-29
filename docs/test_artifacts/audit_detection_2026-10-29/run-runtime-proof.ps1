param(
    [string]$GradleCacheRoot = (Join-Path $env:USERPROFILE '.gradle\caches\modules-2\files-2.1'),
    [string]$AndroidSdkRoot = 'C:\Android\Sdk'
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$compiledProofDir = Join-Path $repoRoot 'android\app\build\validation\audit-detection'
New-Item -ItemType Directory -Path $compiledProofDir -Force | Out-Null
$auditClasspath = @(
    (Join-Path $repoRoot 'android\app\build\tmp\kotlin-classes\debug'),
    (Join-Path $repoRoot 'android\app\build\tmp\kotlin-classes\debugUnitTest')
)
# The app/test classes must already have been built. This script never starts Gradle.
$dependencies = @(
    'org.jetbrains.kotlin\kotlin-stdlib\2.2.20',
    'org.json\json\20260814',
    'org.bouncycastle\bcpkix-jdk18on\1.80',
    'org.bouncycastle\bcprov-jdk18on\1.80.2',
    'org.bouncycastle\bcutil-jdk18on\1.80.2'
)
foreach ($dependency in $dependencies) {
    $jars = @(Get-ChildItem (Join-Path $GradleCacheRoot $dependency) -Recurse -Filter '*.jar')
    if ($jars.Count -ne 1) { throw "Expected one cached runtime jar for $dependency" }
    $auditClasspath += $jars[0].FullName
}
# Real org.json must precede android.jar, whose JSON methods are runtime stubs.
$auditClasspath += Join-Path $AndroidSdkRoot 'platforms\android-36\android.jar'
$classpathString = $auditClasspath -join ';'
& javac -cp $classpathString -d $compiledProofDir (Join-Path $PSScriptRoot 'AuditRuntimeProof.java')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& java -cp ($classpathString + ';' + $compiledProofDir) AuditRuntimeProof $repoRoot 2>&1 |
    Tee-Object (Join-Path $PSScriptRoot 'runtime-proof.txt')
exit $LASTEXITCODE
