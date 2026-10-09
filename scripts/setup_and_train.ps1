param(
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'auto',
    [string]$VisualConfig = 'configs/srm_cnn_genimage.yaml',
    [string]$AudioConfig = 'configs/lcnn_asvspoof.yaml',
    [string]$OutputDir = 'runs/current',
    [string]$DataRoot = '',
    [switch]$InstallDependencies
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit $LASTEXITCODE`: $Executable" }
}

Push-Location -LiteralPath $taskRoot
try {
    if ($InstallDependencies) {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv is required for -InstallDependencies; install it or use an existing Python environment.' }
        # Never replace an existing environment or download datasets implicitly.
        if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
            Invoke-Checked uv @('venv','--python','3.11','.venv')
        }
        Invoke-Checked uv @('pip','install','--python','.venv/Scripts/python.exe','-r','ml/requirements.txt','-r','ml/requirements-export.txt','-r','ml/requirements-data.txt')
        Invoke-Checked uv @('pip','install','--python','.venv/Scripts/python.exe','--no-deps','-e','ml')
    }
    $taskPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Missing .venv. Use -InstallDependencies or run the pipeline with your own configured Python environment.' }
    Push-Location -LiteralPath (Join-Path $taskRoot 'ml')
    try {
        $taskArguments = @('scripts/run_training_pipeline.py','--visual-config',$VisualConfig,'--audio-config',$AudioConfig,'--device',$Device,'--output-dir',$OutputDir)
        if ($DataRoot) { $taskArguments += @('--data-root',$DataRoot) }
        Invoke-Checked $taskPython $taskArguments
    } finally { Pop-Location }
} finally { Pop-Location }
