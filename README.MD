# Prestashop API Search Tool: The Search Detective 🔍

## The Problem I Was Solving

Ever wondered what Prestashop's search API is *actually* returning when you throw 40,000+ product codes at it? Yeah, me
too. This project was born from that exact uncertainty - trying to figure out if customers could actually find products
on my Prestashop site.

After one too many "but it should work!" conversations, I built this tool to get definitive answers about whether
products were findable, and if not, why.

## What This Tool Actually Does

This Python-powered detective interrogates Prestashop's API with your search terms, analyzes what comes back, and tells
you exactly what's happening under the hood. It's like having x-ray vision into Prestashop's search functionality.

The tool will:

- Systematically search your 40,000+ terms against Prestashop
- Track every product match and its reference codes
- Check if each search term appears in the product references it finds (the crucial "is it findable" test)
- Time each search operation (because waiting for Prestashop can feel like watching paint dry)
- Create a comprehensive CSV report with all the data you need

## The Technical Bits

### Core Capabilities

- **Massive Scale Processing**: Handles your 40k+ search terms without breaking a sweat (or your server)
- **Multithreaded Goodness**: Runs multiple searches in parallel because life's too short for sequential API calls
- **Bulletproof Error Handling**: Gracefully manages network issues, API timeouts, and Prestashop's occasional tantrums
- **Memory-Efficient Batch Processing**: Processes data in manageable chunks so your 40k searches don't require 40GB of
  RAM
- **Comprehensive Logging**: Records everything in painful detail, so you know exactly what happened

## Getting This Thing Running

### Quick Setup

1. Download the script (or clone if you're feeling fancy)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Let's Actually Use This Thing

### The Basic Command

```bash
python prestashop_search.py --input your_terms.csv --output results.csv --log search.log --base-url https://your-shop.com --username YOUR_API_KEY
```

### All The Knobs You Can Twist

| Argument     | What It Does                                       | Required? |
|--------------|----------------------------------------------------|-----------|
| `--input`    | Your CSV file with search terms                    | Yes       |
| `--output`   | Where to save the results                          | Yes       |
| `--log`      | Where to dump the play-by-play action              | Yes       |
| `--base-url` | Your Prestashop URL                                | Yes       |
| `--username` | Your API key (used as username in Basic Auth)      | Yes       |
| `--workers`  | How many parallel searches to run (default: 3)     | No        |
| `--limit`    | Stop after X searches (for testing, sanity checks) | No        |

## What Goes In, What Comes Out

### The Input File

Keep it simple: one search term per line in a CSV:

```
product_code_123
another_search_term
that_thing_customers_search_for
```

### The Output Report

You'll get a CSV with these columns:

| Column                | What It Tells You                                         |
|-----------------------|-----------------------------------------------------------|
| `search`              | The term you searched for                                 |
| `quantity_of_results` | How many unique products Prestashop found                 |
| `references`          | The actual product references found                       |
| `include_search`      | TRUE if your search term appears in any of the references |
| `url_checked`         | The exact URL we interrogated                             |
| `date`                | When we ran this search                                   |
| `duration_seconds`    | How long Prestashop took to respond                       |

The `include_search` column is the smoking gun - it tells you whether your search term appears in any of the product
references. If it's FALSE, but you expected TRUE, you've just found a search issue.

## Performance Tuning: Finding The Sweet Spot

### Worker Count Strategy

The `--workers` parameter is crucial. Too few workers and you'll be waiting until the heat death of the universe. Too
many and you might accidentally DDoS your own server (not a great career move).

Some guidelines based on painful experience:

- **Fast, robust server**: 5-7 workers
- **Fragile, temperamental setup**: 2-3 workers
- **Just testing**: Start with 3 and see what breaks

The tool uses a ThreadPoolExecutor for both searching and reference retrieval, so you'll get maximum performance
without hammering your API with too many simultaneous connections.

## Final Thoughts

This tool exists because Prestashop's search functionality is a mysterious black box, and sometimes that box contains
inexplicable behavior that makes customers (and me) frustrated. If you're dealing with thousands of products and
wondering why some aren't showing up in search results, this will give you the data to either fix it or prove it's not
your fault.

Use it wisely, and may your search results always include what they should.
