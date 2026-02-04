# Git Initialization Script for GUVI Submission
# Run this script to prepare your project for Git submission

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Real Estate Investment Advisor" -ForegroundColor Cyan
Write-Host "Git Submission Preparation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Git is installed
Write-Host "Checking Git installation..." -ForegroundColor Yellow
try {
    git --version
    Write-Host "[OK] Git is installed" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Git is not installed. Please install Git first." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 1: Initialize Git Repository" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Initialize Git if not already initialized
if (Test-Path .git) {
    Write-Host "[INFO] Git repository already initialized" -ForegroundColor Yellow
} else {
    git init
    Write-Host "[OK] Git repository initialized" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 2: Configure Git (if needed)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if user name and email are configured
$userName = git config user.name
$userEmail = git config user.email

if ([string]::IsNullOrEmpty($userName)) {
    Write-Host "[INFO] Git user name not configured" -ForegroundColor Yellow
    $name = Read-Host "Enter your name"
    git config user.name "$name"
    Write-Host "[OK] User name configured" -ForegroundColor Green
}

if ([string]::IsNullOrEmpty($userEmail)) {
    Write-Host "[INFO] Git user email not configured" -ForegroundColor Yellow
    $email = Read-Host "Enter your email"
    git config user.email "$email"
    Write-Host "[OK] User email configured" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 3: Review Files to be Committed" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Files that will be included:" -ForegroundColor Yellow
Write-Host "  - README.md" -ForegroundColor White
Write-Host "  - requirements.txt" -ForegroundColor White
Write-Host "  - .gitignore" -ForegroundColor White
Write-Host "  - EDA_Summary.md" -ForegroundColor White
Write-Host "  - SUBMISSION_CHECKLIST.md" -ForegroundColor White
Write-Host "  - streamlit_app.py" -ForegroundColor White
Write-Host "  - src/*.py (3 files)" -ForegroundColor White
Write-Host "  - data/processed/*.png (20 EDA visualizations)" -ForegroundColor White
Write-Host ""
Write-Host "Files that will be excluded (.gitignore):" -ForegroundColor Yellow
Write-Host "  - venv/ (virtual environment)" -ForegroundColor Gray
Write-Host "  - mlruns/ (MLflow runs)" -ForegroundColor Gray
Write-Host "  - *.csv (dataset files)" -ForegroundColor Gray
Write-Host "  - mlflow.db" -ForegroundColor Gray
Write-Host "  - __pycache__/" -ForegroundColor Gray
Write-Host ""

$continue = Read-Host "Continue with Git add? (Y/N)"
if ($continue -ne "Y" -and $continue -ne "y") {
    Write-Host "[INFO] Operation cancelled by user" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 4: Stage Files" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

git add .
Write-Host "[OK] Files staged for commit" -ForegroundColor Green

Write-Host ""
Write-Host "Staged files:" -ForegroundColor Yellow
git status --short

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 5: Commit Changes" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

git commit -m "Initial commit: Real Estate Investment Advisor - GUVI Capstone Project

- Data preprocessing pipeline
- EDA with 20 research questions
- 13 ML models (7 classification + 6 regression)
- MLflow experiment tracking
- Streamlit web application with 5 pages
- Comprehensive documentation"

Write-Host "[OK] Changes committed" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 6: Add Remote Repository" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Enter your GitHub repository URL" -ForegroundColor Yellow
Write-Host "Example: https://github.com/username/repo-name.git" -ForegroundColor Gray
$repoUrl = Read-Host "Repository URL"

if ([string]::IsNullOrEmpty($repoUrl)) {
    Write-Host "[INFO] No repository URL provided. Skipping remote setup." -ForegroundColor Yellow
    Write-Host "[INFO] You can add it later with: git remote add origin <url>" -ForegroundColor Yellow
} else {
    try {
        git remote add origin $repoUrl
        Write-Host "[OK] Remote repository added" -ForegroundColor Green
    } catch {
        Write-Host "[WARNING] Remote 'origin' may already exist" -ForegroundColor Yellow
        Write-Host "[INFO] Use 'git remote set-url origin <url>' to update" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 7: Push to GitHub" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (-not [string]::IsNullOrEmpty($repoUrl)) {
    $push = Read-Host "Push to GitHub now? (Y/N)"
    if ($push -eq "Y" -or $push -eq "y") {
        git branch -M main
        git push -u origin main
        Write-Host "[OK] Code pushed to GitHub" -ForegroundColor Green
    } else {
        Write-Host "[INFO] Push skipped. Run manually: git push -u origin main" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Git Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Verify your GitHub repository" -ForegroundColor White
Write-Host "2. Ensure repository is PUBLIC" -ForegroundColor White
Write-Host "3. Submit GitHub URL to GUVI" -ForegroundColor White
Write-Host "4. Review SUBMISSION_CHECKLIST.md" -ForegroundColor White
Write-Host ""
Write-Host "Project is ready for submission! 🎉" -ForegroundColor Green
Write-Host ""
