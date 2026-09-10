# Planted Conflicts (for testing conflict detection accuracy)

NOTE: Conflicts #1-3 (return window, refund time, sale items) were resolved in the FAQ
as of August 2026. Conflict #4 (order cancellation window) was planted to replace them
for ongoing conflict-detection testing.

These are GENUINE, unambiguous contradictions between official policy documents
and `03_website_faq.txt` (live site copy that drifted out of date). Use this list to
check precision/recall once conflict detection is built — it should catch conflict #4
(the only currently active one) and should NOT flag anything else as a false positive.

1. **Return window**
   - Official: 30 days from delivery
   - FAQ: 14 days from delivery
   - → Genuine conflict, same policy, different numbers.

2. **Refund processing time**
   - Official: 5-7 business days
   - FAQ: 3-5 business days
   - → Genuine conflict, same policy, different numbers.

3. **Sale item returns**
   - Official: Store credit eligible (unless defective)
   - FAQ: "Final sale, cannot be returned or exchanged" (no exception mentioned)
   - → Genuine conflict, contradicts on eligibility itself.

4. **Order cancellation window**
   - Official: 1 hour from order placement
   - FAQ: 24 hours from order placement
   - → Genuine conflict, same policy, different numbers.
## Non-conflicts to check the system does NOT flag
- Shipping free-over-$75 threshold appears identically in both `01_shipping_policy.txt`
  and `03_website_faq.txt` — should NOT be flagged (they agree).
- Warranty (footwear 6mo / apparel 3mo) vs "damaged on arrival" 7-day window in
  `04_warranty_policy.txt` — these are different scenarios (defect vs shipping damage),
  not a conflict, even though both are about "damaged" items.
