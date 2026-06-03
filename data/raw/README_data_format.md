# Raw Data Format

This project does not download datasets automatically. It does not use the Kaggle API, Hugging Face datasets, web scraping, or external APIs. The student must manually download the required Kaggle CSV files and place them in this folder.

## Expected Files

Main source:

- Kaggle slug: `austinreese/craigslist-carstrucks-data`
- Original file: `vehicles.csv`
- Copy or rename to: `data/raw/kaggle_craigslist_vehicles.csv`

Secondary source:

- Kaggle slug: `andreinovikov/used-cars-dataset`
- Copy the downloaded CSV to: `data/raw/kaggle_carscom_used_cars.csv`

## Internal Schema

The code maps raw files into:

- `listing_id`
- `source_name`
- `brand`
- `model`
- `year`
- `mileage_km`
- `fuel_type`
- `transmission`
- `body_type`
- `engine_size_l`
- `condition`
- `title_status`
- `drive`
- `paint_color`
- `location_country`
- `location_region`
- `currency`
- `seller_price`
- `seller_description`

## Craigslist Mapping

- `listing_id` <- `id`
- `source_name` <- fixed value `kaggle_craigslist`
- `brand` <- `manufacturer`
- `model` <- `model`
- `year` <- `year`
- `mileage_km` <- `odometer * 1.60934`
- `fuel_type` <- `fuel`
- `transmission` <- `transmission`
- `body_type` <- `type`
- `engine_size_l` <- empty
- `condition` <- `condition`
- `title_status` <- `title_status`
- `drive` <- `drive`
- `paint_color` <- `paint_color`
- `location_country` <- fixed value `USA`
- `location_region` <- `state`, otherwise `region`
- `currency` <- fixed value `USD`
- `seller_price` <- `price`
- `seller_description` <- `description`

The Craigslist `odometer` column is treated as miles and converted to kilometers.

## Cars.com Mapping

The Cars.com dataset may use different column names. The project detects compatible columns safely. Possible names include:

- brand: `brand`, `make`, `manufacturer`
- model: `model`
- year: `model_year`, `year`
- mileage: `milage`, `mileage`, `odometer`
- price: `price`, `list_price`, `seller_price`
- fuel: `fuel_type`, `fuel`
- body: `body_type`, `body`, `type`
- title: `accident`, `clean_title`, `title_status`
- engine: `engine`, `engine_size`
- color: `exterior_color`, `paint_color`

If a compatible secondary column is not found, the code fills `NaN` or `Unknown`, prints a warning, and continues when enough critical fields exist. It does not invent missing data.

## Required and Optional Columns

Critical fields for training are:

- `seller_price`
- `year`
- `mileage_km`
- `brand`
- `model`

Useful optional fields include:

- `fuel_type`
- `transmission`
- `body_type`
- `condition`
- `title_status`
- `drive`
- `paint_color`
- `location_region`
- `seller_description`

Craigslist descriptions are used for NLP. Cars.com descriptions may be empty and are still usable for numeric ML when structured fields are available.

## Demo Fallback

If both real raw CSV files are missing, the project creates a tiny DEMO ONLY fallback dataset so scripts and the app can launch. Console output will print:

`WARNING: Real Kaggle CSV files not found. Using DEMO ONLY fallback data. This is not valid for final submission.`

The fallback data is not valid for final university submission.

