$ErrorActionPreference = "Stop"

$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$DatasetDir = if ($env:DATASET_DIR) { $env:DATASET_DIR } else { "runs\office" }
$SemanticIds = if ($env:SEMANTIC_IDS) { $env:SEMANTIC_IDS } else { "runs\office\semantic_ids_rq.json" }
$BaseOutputDir = if ($env:BASE_OUTPUT_DIR) { $env:BASE_OUTPUT_DIR } else { "runs\office" }
$Device = if ($env:DEVICE) { $env:DEVICE } else { "cuda" }

$Epochs = if ($env:EPOCHS) { $env:EPOCHS } else { "100" }
$EarlyStopPatience = if ($env:EARLY_STOP_PATIENCE) { $env:EARLY_STOP_PATIENCE } else { "10" }
$BatchSize = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { "256" }
$MaxLen = if ($env:MAX_LEN) { $env:MAX_LEN } else { "50" }
$Dim = if ($env:DIM) { $env:DIM } else { "128" }
$NumRandomNeg = if ($env:NUM_RANDOM_NEG) { $env:NUM_RANDOM_NEG } else { "100" }

function Run-Exp {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Name,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]] $ExtraArgs
    )

    $OutputDir = Join-Path $BaseOutputDir $Name
    Write-Host "============================================================"
    Write-Host "Running: $Name"
    Write-Host "Output: $OutputDir"
    Write-Host "============================================================"

    & $PythonBin scripts\train_qsdrec.py `
        --dataset-dir $DatasetDir `
        --semantic-ids $SemanticIds `
        --output-dir $OutputDir `
        --device $Device `
        --epochs $Epochs `
        --early-stop-patience $EarlyStopPatience `
        --batch-size $BatchSize `
        --max-len $MaxLen `
        --dim $Dim `
        --num-random-neg $NumRandomNeg `
        @ExtraArgs
}

# 1. Pure SASRec baseline.
Run-Exp exp_sasrec `
    --num-interests 1 `
    --num-hard-neg 0 `
    --sem-weight 0 `
    --dis-weight 0 `
    --div-weight 0

# 2. Semantic fusion weight.
Run-Exp exp_sem005 `
    --num-interests 4 `
    --num-hard-neg 0 `
    --sem-weight 0.05 `
    --dis-weight 0 `
    --div-weight 0

Run-Exp exp_sem010 `
    --num-interests 4 `
    --num-hard-neg 0 `
    --sem-weight 0.10 `
    --dis-weight 0 `
    --div-weight 0

Run-Exp exp_sem020 `
    --num-interests 4 `
    --num-hard-neg 0 `
    --sem-weight 0.20 `
    --dis-weight 0 `
    --div-weight 0

Run-Exp exp_sem050 `
    --num-interests 4 `
    --num-hard-neg 0 `
    --sem-weight 0.50 `
    --dis-weight 0 `
    --div-weight 0

Run-Exp exp_sem100 `
    --num-interests 4 `
    --num-hard-neg 0 `
    --sem-weight 1.00 `
    --dis-weight 0 `
    --div-weight 0

# 3. Multi-interest query.
Run-Exp exp_interest1_sem010 `
    --num-interests 1 `
    --num-hard-neg 0 `
    --sem-weight 0.10 `
    --dis-weight 0 `
    --div-weight 0

Run-Exp exp_interest2_sem010 `
    --num-interests 2 `
    --num-hard-neg 0 `
    --sem-weight 0.10 `
    --dis-weight 0 `
    --div-weight 0

Run-Exp exp_interest4_sem010 `
    --num-interests 4 `
    --num-hard-neg 0 `
    --sem-weight 0.10 `
    --dis-weight 0 `
    --div-weight 0

Run-Exp exp_interest8_sem010 `
    --num-interests 8 `
    --num-hard-neg 0 `
    --sem-weight 0.10 `
    --dis-weight 0 `
    --div-weight 0

# 4. Disambiguation loss.
Run-Exp exp_dis002 `
    --num-interests 4 `
    --num-hard-neg 0 `
    --sem-weight 0.10 `
    --dis-weight 0.02 `
    --div-weight 0

Run-Exp exp_dis005 `
    --num-interests 4 `
    --num-hard-neg 0 `
    --sem-weight 0.10 `
    --dis-weight 0.05 `
    --div-weight 0

Run-Exp exp_dis010 `
    --num-interests 4 `
    --num-hard-neg 0 `
    --sem-weight 0.10 `
    --dis-weight 0.10 `
    --div-weight 0

Run-Exp exp_dis020 `
    --num-interests 4 `
    --num-hard-neg 0 `
    --sem-weight 0.10 `
    --dis-weight 0.20 `
    --div-weight 0

# 5. Diversity loss.
Run-Exp exp_div001 `
    --num-interests 4 `
    --num-hard-neg 0 `
    --sem-weight 0.10 `
    --dis-weight 0.05 `
    --div-weight 0.001

Run-Exp exp_div005 `
    --num-interests 4 `
    --num-hard-neg 0 `
    --sem-weight 0.10 `
    --dis-weight 0.05 `
    --div-weight 0.005

Run-Exp exp_div010 `
    --num-interests 4 `
    --num-hard-neg 0 `
    --sem-weight 0.10 `
    --dis-weight 0.05 `
    --div-weight 0.010

# 6. Prefix hard negatives.
Run-Exp exp_hard5_p2 `
    --num-interests 4 `
    --prefix-level 2 `
    --num-hard-neg 5 `
    --sem-weight 0.10 `
    --dis-weight 0.05 `
    --div-weight 0.005

Run-Exp exp_hard10_p2 `
    --num-interests 4 `
    --prefix-level 2 `
    --num-hard-neg 10 `
    --sem-weight 0.10 `
    --dis-weight 0.05 `
    --div-weight 0.005

Run-Exp exp_hard20_p2 `
    --num-interests 4 `
    --prefix-level 2 `
    --num-hard-neg 20 `
    --sem-weight 0.10 `
    --dis-weight 0.05 `
    --div-weight 0.005

Run-Exp exp_hard10_p1 `
    --num-interests 4 `
    --prefix-level 1 `
    --num-hard-neg 10 `
    --sem-weight 0.10 `
    --dis-weight 0.05 `
    --div-weight 0.005

Run-Exp exp_hard10_p3 `
    --num-interests 4 `
    --prefix-level 3 `
    --num-hard-neg 10 `
    --sem-weight 0.10 `
    --dis-weight 0.05 `
    --div-weight 0.005
