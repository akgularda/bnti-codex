# bnti
Border Neighbours Threat Index

<div>
NOTE: This program does not have any kind of political meaning. It only is used as educational purposes.
</div>

Border Neighbours Threat Index is an index that analyzes countries' news and classify if they are negative.

**Improvements in this version:**
- **Zero-Shot Classification**: Now uses `facebook/bart-large-mnli` to detect specific threats ("military threat", "political conflict", "economic crisis") rather than just "sentiment".
- **Performance**: Fetches RSS feeds in parallel, significantly speeding up the process.
- **Refined Logic**: Includes logic for NATO alliances and more accurate scoring.

## Dependencies

You need to install the required Python packages:

```bash
pip install -r requirements.txt
```

*(Note for GPU users: You may want to install PyTorch manually from pytorch.org for CUDA support, but the standard install works for CPU)*

## ☁️ Cloud Deployment (Free Automation)

You can run BNTI "forever" in the cloud without keeping your PC on.
See the **[Deployment Guide](DEPLOYMENT_GUIDE.md)** for 3 simple steps to activate this.

It uses **GitHub Actions** (included) to:
1.  Run the analysis every 6 hours.
2.  Update the dashboard automatically.
3.  Host the site on GitHub Pages.

## Usage

Run the script directly:
```bash
python borderneighboursthreatindex.py
```

The program will:
1.  Download the AI model (approx 1.5GB) on the first run.
2.  Fetch news from defined RSS feeds.
3.  Analyze headlines for threats.
4.  Save reports and an Excel file to your **Desktop**.

<div align="center">
<img width="575" alt="image" src="bnti.png">
</div>

<div>
This program uses open-source AI models to analyze geopolitical tensions based on public news feeds.
</div>
