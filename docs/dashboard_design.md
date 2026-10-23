# Dashboard Design Summary
**Documented: 02/09/2026, prior to implementation**

## Architecture
All station data, SARIMA models (53), and LSTM models (51) are loaded into memory once at
app startup (see `dashboard_architecture.png`). User interactions (map clicks, tab switches,
model toggle, horizon slider) are handled by Dash callbacks that read from this pre-loaded
state — no re-fitting or re-querying occurs per interaction, satisfying NFR5 and supporting
NFR1/NFR2's load-time and responsiveness targets.

## Station line ordering methodology
The dashboard's schematic map requires stations to be sequenced correctly along each line —
this is not present in the source dataset. Ordering was derived via a nearest-neighbour walk
on real GPS coordinates (starting from each line's known terminus), validated against RTA's
officially published network map. Where the data-driven method and the official map
disagreed (the Route 2020 branch point at Jabal Ali, where straight-line GPS distance
does not reflect the branching rail topology), the official map's topology was used, since
schematic transit maps prioritise connectivity accuracy over literal distance. This process
also surfaced and helped diagnose a data-integrity bug: Burj Khalifa/Dubai Mall Metro Station
was found to be missing its line and coordinate data entirely, traced to a station-name
formatting inconsistency during an earlier data-cleaning phase, and was corrected at the
source before the final map was built.

## Interactive prototype
A working HTML/JS prototype validated the full interaction model before implementation:
clicking any of the 53 stations (via schematic map or dropdown) updates a detail panel
showing historical trend, a model toggle (SARIMA/LSTM) with a forecast confidence interval,
an adjustable forecast horizon, and a recommendation badge reflecting the actual
holdout-validated MAPE comparison for that station. The prototype also modelled NFR3's
required graceful-degradation behaviour: for the two stations with no fitted LSTM model,
the toggle disables with an explanatory message rather than erroring.

## Design decisions confirmed
- Map: schematic (stylised line diagram), not a real-world basemap
- Map colour: by line, not ridership volume
- Model display: toggle (one shown at a time), not overlay
- Scope: single-station focus; station-vs-station comparison out of scope