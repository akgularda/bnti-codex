# BNTI Cloud Deployment Guide (Zero Cost)

This system is set up to run automatically on **GitHub Actions**.
It will:
1.  Run the Python Intelligence Analyzer every 6 hours.
2.  Update the `bnti_data.js` file.
3.  Publish the updated Dashboard (`index.html`) to various devices via a public URL.

## Step 1: Create a GitHub Repository
1.  Go to **[GitHub.com](https://github.com/new)** and sign in.
2.  Create a **New Repository**.
    *   Name: `bnti-live` (or anything you want).
    *   Public/Private: **Public** (Required for free GitHub Pages).
    *   **Do not** initialize with README/gitignore (you already have them).

## Step 2: Upload Files
You have two options:

### Option A: Using Desktop App (Easiest)
1.  Download **GitHub Desktop**.
2.  File > Add Local Repository > Select your `bnti-main` folder (`C:\Users\akgul\Downloads\bnti-main`).
3.  Click "Publish Repository" to push it to the GitHub repo you created.

### Option B: Using Command Line
Open a terminal in your folder (`C:\Users\akgul\Downloads\bnti-main`) and run:
```bash
git init
git add .
git commit -m "Initial BNTI Intelligence Deploy"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/bnti-live.git
git push -u origin main
```
*(Replace `YOUR_USERNAME` with your actual GitHub username)*

## Step 3: Enable Automation (Pages)
1.  Go to your repository on GitHub.
2.  Click **Settings** (top right tab).
3.  On the left sidebar, click **Pages**.
4.  Under **Build and deployment**:
    *   Source: **GitHub Actions** (This is crucial! Do not select "Deploy from branch").
5.  Go to the **Actions** tab (top of page).
    *   You should see "BNTI Intelligence Update" running (or waiting to run).
    *   If not, you can click "BNTI Intelligence Update" on the left -> "Run workflow".

## Step 4: View Your Live Dashboard
Once the "Deploy Dashboard" action finishes (takes ~2-3 mins first time):
*   Your site will be live at: `https://YOUR_USERNAME.github.io/bnti-live/`
*   You can open this link on your **phone**, tablet, or any PC.
*   It updates automatically every 6 hours.
