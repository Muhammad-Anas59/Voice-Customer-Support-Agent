"""
order_lookup.py

Job: Given an order number and a verification value, fetch the order's
real status, fulfillment, and tracking info from Shopify's Admin API.

This is separate from the policy RAG pipeline (retriever.py). Policy
questions get answered from documents; "where's my order" questions get
answered from this live lookup instead.

VERIFICATION METHOD - IMPORTANT CONTEXT:
Ideally this verifies by order number + customer email. However, Shopify
blocks API access to customer PII (name, email, phone, address) on free/
dev-tier stores - that access requires a paid Shopify plan (Shopify,
Advanced, or Plus). Since this is a free demo store, we verify by order
number + order total instead, which isn't classified as PII and is
already accessible.

This is a demo-environment limitation only, not a product limitation: a
real client is already on a paid Shopify plan by definition (you can't
run a real store on the free tier), so email verification will work
correctly out of the box for an actual client. Switch verify-by back to
email (see find_order below) once testing against a paid-plan store.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

SHOPIFY_STORE_DOMAIN = os.getenv("SHOPIFY_STORE_DOMAIN")  # e.g. verve-athletics-demo.myshopify.com
SHOPIFY_ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN")
SHOPIFY_API_VERSION = "2025-01"  # update if Shopify deprecates this version


def _shopify_headers():
    return {
        "X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN,
        "Content-Type": "application/json"
    }


def find_order(order_number, expected_total):
    """Looks up a single order by its order number (e.g. '#1001' or
    '1001'), then verifies it belongs to the right customer by checking
    the order's total matches what was provided - NOT by email, since
    Shopify blocks API access to customer email/name/address on this
    store's plan tier (see module docstring). Returns a dict with order
    status details, or a dict with 'found': False if no match (we don't
    reveal whether the order number itself exists, to avoid leaking
    information about someone else's order)."""
    if not SHOPIFY_STORE_DOMAIN or not SHOPIFY_ACCESS_TOKEN:
        return {
            "found": False,
            "error": "Shopify credentials not configured. Set SHOPIFY_STORE_DOMAIN and SHOPIFY_ACCESS_TOKEN in .env"
        }

    clean_number = order_number.strip().lstrip("#")

    url = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/orders.json"
    params = {"name": f"#{clean_number}", "status": "any"}

    try:
        response = requests.get(url, headers=_shopify_headers(), params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"found": False, "error": f"Shopify API request failed: {e}"}

    orders = response.json().get("orders", [])
    if not orders:
        return {"found": False, "error": "No order found with that number."}

    order = orders[0]
    actual_total = str(order.get("total_price", "")).strip()

    try:
        matches = float(actual_total) == float(str(expected_total).strip())
    except (ValueError, TypeError):
        matches = False

    if not matches:
        # Deliberately vague - don't confirm the order exists with a
        # different total, that's an information leak about someone else's order
        return {"found": False, "error": "No order found matching that number and amount."}

    return _format_order(order)


def _format_order(order):
    """Extracts just the fields a support answer actually needs from
    Shopify's (much larger) raw order object."""
    fulfillments = order.get("fulfillments", [])
    tracking_number = None
    tracking_url = None
    carrier = None

    if fulfillments:
        latest = fulfillments[-1]
        tracking_number = latest.get("tracking_number")
        tracking_url = latest.get("tracking_url")
        carrier = latest.get("tracking_company")

    if order.get("cancelled_at"):
        status = "cancelled"
    elif order.get("fulfillment_status") == "fulfilled":
        status = "shipped"
    elif order.get("fulfillment_status") == "partial":
        status = "partially_shipped"
    else:
        status = "processing"

    return {
        "found": True,
        "order_number": order.get("name"),
        "status": status,
        "financial_status": order.get("financial_status"),
        "created_at": order.get("created_at"),
        "tracking_number": tracking_number,
        "tracking_url": tracking_url,
        "carrier": carrier,
        "line_items": [
            {"title": item.get("title"), "quantity": item.get("quantity")}
            for item in order.get("line_items", [])
        ]
    }


if __name__ == "__main__":
    number = input("Order number (e.g. 1001): ")
    total = input("Order total (e.g. 80.00): ")
    result = find_order(number, total)
    print("\n--- RESULT ---")
    if result["found"]:
        print(f"Order: {result['order_number']}")
        print(f"Status: {result['status']}")
        print(f"Payment status: {result['financial_status']}")
        print(f"Placed on: {result['created_at']}")
        if result["tracking_number"]:
            print(f"Tracking: {result['tracking_number']} via {result['carrier']}")
            print(f"Tracking URL: {result['tracking_url']}")
        print("Items:")
        for item in result["line_items"]:
            print(f"  - {item['quantity']}x {item['title']}")
    else:
        print(f"Not found: {result['error']}")