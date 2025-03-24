#!/usr/bin/env python3
"""
Prestashop API Search Tool

This script searches a Prestashop site via API using search criteria from a CSV file
and exports results to a CSV with detailed information about matches.

Authentication is handled using Basic Auth with the provided username.
"""

import csv
import logging
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import quote

import requests

try:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover
    pass  # pylint: disable=import-error


class PrestashopClient:
    """Client for interacting with Prestashop API."""

    def __init__(self, base_url: str, username: str, logger: logging.Logger):
        """
        Initialize the Prestashop API client.

        Args:
            base_url: API base URL
            username: API username for Basic Auth
            logger: Logger instance
        """
        self.base_url = base_url
        self.session = self._create_session(username)
        self.logger = logger

    @staticmethod
    def _create_session(username: str) -> requests.Session:
        """
        Create a requests session with retry logic and basic auth.

        Args:
            username: API username for Basic Auth

        Returns:
            Configured requests session
        """
        session = requests.Session()
        session.auth = (username, "")

        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def search_products(self, search_term: str) -> Tuple[List[str], str, float]:
        """
        Search for products by term and return product IDs.

        Args:
            search_term: Search criteria

        Returns:
            Tuple containing:
            - List of product IDs
            - URL that was checked
            - Time taken for the request

        Raises:
            requests.RequestException: For HTTP errors
            ValueError: For authentication or parsing errors
        """
        encoded_term = quote(search_term)
        url = f"{self.base_url}/api/search?language=1&query={encoded_term}"

        start_time = time.time()
        try:
            self.logger.info("Searching for: %s", search_term)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            if "authentication" in response.text.lower() and "error" in response.text.lower():
                self.logger.error("Authentication error for search '%s'. Check API credentials.", search_term)
                raise ValueError("API authentication failed")

            root = ET.fromstring(response.text)
            product_ids = []

            for product in root.findall(".//product"):
                product_ids.append(product.get("id"))

            time_taken = time.time() - start_time
            # Log the results
            products_count = len(product_ids)
            self.logger.info(
                "Found %d products for '%s' in %.2f seconds",
                products_count, search_term, time_taken
            )
            return product_ids, url, time_taken

        except (requests.RequestException, ET.ParseError) as err:
            time_taken = time.time() - start_time
            self.logger.error("Error searching for '%s': %s", search_term, str(err))
            return [], url, time_taken

    def get_product_reference(self, product_id: str) -> Optional[str]:
        """
        Get product reference by ID.

        Args:
            product_id: Product ID

        Returns:
            Product reference or None if not found
        """
        url = f"{self.base_url}/api/products/{product_id}"

        try:
            self.logger.debug("Getting details for product ID: %s", product_id)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.text)
            reference_elem = root.find(".//reference")

            if reference_elem is not None and reference_elem.text:
                # Strip CDATA if present
                reference = reference_elem.text
                if "![CDATA[" in reference:
                    reference = reference.replace("![CDATA[", "").replace("]]", "").strip()
                self.logger.debug("Product %s reference: %s", product_id, reference)
                return reference

            self.logger.warning("No reference found for product ID: %s", product_id)
            return None

        except (requests.RequestException, ET.ParseError) as err:
            self.logger.error("Error getting product reference for ID %s: %s", product_id, str(err))
            return None


class SearchProcessor:
    """Processor for handling search operations and results."""

    def __init__(self, api_client: PrestashopClient, logger: logging.Logger):
        """
        Initialize the search processor.

        Args:
            api_client: Prestashop API client instance
            logger: Logger instance
        """
        self.api_client = api_client
        self.logger = logger

    @staticmethod
    def validate_search_results(references: List[str], search_term: str) -> bool:
        """
        Validate if search term appears in any references.

        Args:
            references: List of product references
            search_term: Original search term

        Returns:
            True if search term is found in any reference, False otherwise
        """
        if not references or not search_term:
            return False

        search_lowercase = search_term.lower()
        return any(search_lowercase in ref.lower() for ref in references)

    def process_search_term(self, search_term: str) -> Dict[str, Union[str, int, float, bool]]:
        """
        Process a single search term and return results.

        Args:
            search_term: Search criteria

        Returns:
            Dictionary containing search results
        """
        search_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            product_ids, url_checked, search_duration = self.api_client.search_products(search_term)

            references = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                reference_futures = {
                    executor.submit(self.api_client.get_product_reference, product_id): product_id
                    for product_id in product_ids
                }

            for reference_future in reference_futures:
                try:
                    reference = reference_future.result()
                    if reference:
                        references.append(reference)
                except Exception as err:  # pylint: disable=broad-exception-caught
                    product_id = reference_futures[reference_future]
                    self.logger.error(
                        "Error processing product %s for search '%s': %s",
                        product_id, search_term, str(err)
                    )

            unique_references = list(set(references))
            include_search = self.validate_search_results(unique_references, search_term)

            unique_product_count = len(set(product_ids))

            # Prepare result
            result = {
                "search": search_term,
                "quantity_of_results": unique_product_count,
                "references": ", ".join(unique_references),
                "include_search": include_search,
                "url_checked": url_checked,
                "date": search_time,
                "duration_seconds": round(search_duration, 3)
            }

            self.logger.info(
                "Completed search for '%s': %d unique products found",
                search_term, unique_product_count
            )
            return result

        except Exception as err:  # pylint: disable=broad-exception-caught
            self.logger.error("Error in process_search_term for '%s': %s", search_term, str(err))
            url = f"{self.api_client.base_url}/api/search?language=1&query={search_term}"
            return {
                "search": search_term,
                "quantity_of_results": 0,
                "references": "",
                "include_search": False,
                "url_checked": url,
                "date": search_time,
                "duration_seconds": 0.0
            }


class SearchConfig:
    """Configuration for search operations."""

    def __init__(
            self,
            input_csv: str,
            output_csv: str,
            max_workers: int,
            limit: Optional[int]
    ):
        """
        Initialize search configuration.

        Args:
            input_csv: Path to input CSV file
            output_csv: Path to output CSV file
            max_workers: Maximum number of concurrent workers
            limit: Optional limit for number of terms to process
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.max_workers = max_workers
        self.limit = limit


class SearchJobRunner:
    """Runs search jobs in batches with reporting."""

    def __init__(
            self,
            processor: SearchProcessor,
            config: SearchConfig,
            logger: logging.Logger
    ):
        """
        Initialize the search job runner.

        Args:
            processor: Search processor instance
            config: Search configuration
            logger: Logger instance
        """
        self.processor = processor
        self.config = config
        self.logger = logger
        self.fieldnames = [
            "search",
            "quantity_of_results",
            "references",
            "include_search",
            "url_checked",
            "date",
            "duration_seconds"
        ]

    def load_search_terms(self) -> List[str]:
        """
        Load search terms from input CSV.

        Returns:
            List of search terms

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If no search terms found
        """
        search_terms = []
        try:
            with open(self.config.input_csv, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row and row[0].strip():
                        search_terms.append(row[0].strip())

            if not search_terms:
                raise ValueError("No search terms found in the input CSV file.")

            self.logger.info(
                "Loaded %d search terms from %s",
                len(search_terms),
                self.config.input_csv
            )

            if self.config.limit and self.config.limit < len(search_terms):
                self.logger.info("Limiting to first %d search terms", self.config.limit)
                search_terms = search_terms[:self.config.limit]

            return search_terms

        except (FileNotFoundError, ValueError) as err:
            self.logger.error("Error reading input CSV file: %s", str(err))
            raise

    def process_batch(self, batch: List[str]) -> List[Dict]:
        """
        Process a batch of search terms.

        Args:
            batch: List of search terms to process

        Returns:
            List of result dictionaries
        """
        batch_results = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self.processor.process_search_term, term): term
                for term in batch
            }

            for future in futures:
                try:
                    result = future.result()
                    batch_results.append(result)

                except Exception as err:  # pylint: disable=broad-exception-caught
                    term = futures[future]
                    self.logger.error("Unhandled error processing term '%s': %s", term, str(err))

                    api_base = self.processor.api_client.base_url
                    encoded_term = quote(term)
                    search_url = f"{api_base}/api/search?language=1&query={encoded_term}"

                    batch_results.append({
                        "search": term,
                        "quantity_of_results": 0,
                        "references": "",
                        "include_search": False,
                        "url_checked": search_url,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "duration_seconds": 0.0
                    })

        return batch_results

    def run(self) -> None:
        """
        Run the search job with batch processing.
        """
        try:
            search_terms = self.load_search_terms()
        except (FileNotFoundError, ValueError):
            return

        output_path = Path(self.config.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_processed = 0
        total_products_found = 0
        total_duration = 0.0
        terms_with_included_search = 0

        with open(self.config.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

            batch_size = 100
            total_batches = (len(search_terms) + batch_size - 1) // batch_size

            for i in range(0, len(search_terms), batch_size):
                batch = search_terms[i:i + batch_size]
                current_batch = i // batch_size + 1
                self.logger.info("Processing batch %d/%d", current_batch, total_batches)

                # Process batch
                batch_results = self.process_batch(batch)

                # Update statistics
                for result in batch_results:
                    total_products_found += result["quantity_of_results"]
                    total_duration += result["duration_seconds"]
                    if result["include_search"]:
                        terms_with_included_search += 1

                writer.writerows(batch_results)
                csvfile.flush()

                total_processed += len(batch_results)

                self.logger.info(
                    "Completed batch %d, processed %d/%d terms",
                    current_batch, total_processed, len(search_terms)
                )

        self.logger.info("Processing complete. Results written to %s", self.config.output_csv)

        avg_duration = total_duration / total_processed if total_processed else 0

        self.logger.info("Summary:")
        self.logger.info("- Total search terms processed: %d", total_processed)
        self.logger.info("- Total unique products found: %d", total_products_found)
        self.logger.info("- Average search duration: %.3f seconds", avg_duration)
        self.logger.info(
            "- Terms with products that include search criteria: %d",
            terms_with_included_search
        )


def setup_logging(log_file: str) -> logging.Logger:
    """
    Set up and configure logger with file and console output.

    Args:
        log_file: Path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("prestashop_search")
    logger.setLevel(logging.INFO)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Prestashop API Search Tool")
    parser.add_argument("--input", required=True, help="Input CSV file with search terms")
    parser.add_argument("--output", required=True, help="Output CSV file for results")
    parser.add_argument("--log", required=True, help="Log file path")
    parser.add_argument("--base-url", required=True, help="Prestashop API base URL")
    parser.add_argument("--username", required=True, help="API username for Basic Authentication")
    parser.add_argument(
        "--workers", type=int, default=3, help="Maximum number of concurrent workers"
    )
    parser.add_argument("--limit", type=int, help="Limit number of search terms to process")

    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""

    args = parse_args()

    logger = setup_logging(args.log)
    logger.info("Starting Prestashop search with input file: %s", args.input)

    client = PrestashopClient(args.base_url, args.username, logger)

    processor = SearchProcessor(client, logger)

    config = SearchConfig(
        args.input,
        args.output,
        args.workers,
        args.limit
    )

    job_runner = SearchJobRunner(processor, config, logger)
    job_runner.run()


if __name__ == "__main__":
    main()
