# Google Maps Reviews Analyzer

A Python tool for collecting and analyzing Google Maps reviews using BERT sentiment analysis.

## Features

- Collect reviews from Google Maps API for specific locations
- Support for multiple languages (Chinese and English)
- Related places discovery within a specified radius
- BERT-based sentiment analysis for review classification
- Batch processing for efficient analysis
- Automatic GPU acceleration when available
- CSV export with sentiment analysis results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/google-maps-reviews-analyzer.git
cd google-maps-reviews-analyzer
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Google Maps API key:
```
GOOGLE_MAPS_API_KEY=your_api_key_here
```

## Usage

### Collecting Reviews

```bash
# Basic usage (uses default Place ID)
python more_reviews.py

# Search by place name
python more_reviews.py --query "台北君悅酒店"

# Collect reviews from related places
python more_reviews.py --related --max_related 5

# Specify output directory
python more_reviews.py --output_dir "my_data_folder"
```

### Analyzing Reviews

```bash
python analyze_reviews.py
```

The script will:
1. Find the most recent CSV file in the data_store directory
2. Perform sentiment analysis using BERT
3. Save results to a new CSV file
4. Display a summary of sentiment distribution

## Output

The analysis generates a CSV file with the following columns:
- place_name: Name of the location
- place_id: Google Maps Place ID
- place_rating: Overall rating of the place
- author_name: Name of the reviewer
- rating: Individual review rating
- language: Language of the review
- review_text: The review content
- time: Timestamp of the review
- relative_time: Relative time description
- sentiment: BERT sentiment analysis result (positive/neutral/negative)

## Requirements

- Python 3.7+
- Google Maps API key
- Required Python packages (see requirements.txt)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 

You can use 
curl "https://maps.googleapis.com/maps/api/place/textsearch/json?query=Grand+Hyatt+Taipei&language=zh-TW&key=Your_Key"
to get the google maps api's ID