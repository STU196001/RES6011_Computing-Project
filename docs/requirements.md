# Dubai Metro Ridership Dashboard — Requirements

## Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR1 | User can view a map of all 53 Dubai Metro stations, colour-coded by line (Red/Green/Route 2020/Interchange) | Must |
| FR2 | User can select a station by clicking its map marker, or via a dropdown/search as an alternative input method | Must |
| FR3 | User can view historical monthly ridership (Jan 2018–Aug 2023) for the selected station | Must |
| FR4 | User can view a forecast (1–12 months, adjustable via slider) for the selected station | Must |
| FR5 | User can toggle between SARIMA and LSTM forecasts for the selected station | Must |
| FR6 | User can view the forecast's confidence interval alongside the SARIMA prediction | Should |
| FR7 | User can see which model (SARIMA/LSTM) performed better historically for the selected station, with the supporting MAPE figures | Should |
| FR8 | User can view a full comparison table of SARIMA vs LSTM performance across all 53 stations, sortable by column | Should |
| FR9 | User can view network-level insights: system-wide trend, seasonal pattern, and notable station-specific findings (EXPO, Route 2020, World Trade Centre) | Could |

## Non-Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| NFR1 | Initial dashboard load completes in under 3 seconds | Must |
| NFR2 | Station selection updates the detail panel within 2 seconds | Must |
| NFR3 | Dashboard handles stations with no LSTM model (2 of 53) gracefully — toggle disables with an explanatory note, no crash | Must |
| NFR4 | Layout remains usable without horizontal scrolling on a 1366×768 screen | Should |
| NFR5 | All models and station data are pre-loaded into memory at app startup, not re-fitted or re-queried per interaction | Must |

## Users

- **Primary**: transport planners/analysts wanting station-level ridership forecasts to inform service planning
- **Secondary**: general public/students wanting to explore Dubai Metro ridership patterns and understand forecasting model trade-offs

## Design Decisions (documented 02/09/2026)

- **Map colour encoding**: by line, not ridership volume — prioritises network structure clarity over a heatmap, which would need frequent rescaling as forecasts update
- **Model display**: toggle (one model shown at a time) rather than overlay — reduces visual clutter given confidence intervals are also displayed
- **Scope**: single-station focus; station-vs-station comparison explicitly out of scope for this version