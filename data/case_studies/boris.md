# Case study: boris

> Storm Boris brought extreme rainfall to Central Europe in mid-September 2024: between 12 and 16 September, more than 300 mm of rain fell in the town of Jeseník, Czechia. The Oder river at Opole, Poland rose to its highest level in over two decades, and flooding spread across Poland, Czechia, Austria and Slovakia. In Vienna, Austria, the storm brought wind gusts of over 150 km/h.

## Claims
- **FLOOD · Jeseník, Czechia · 2024-09-12**
  - Storm Boris brought more than 300 mm of rainfall to the town of Jeseník, Czechia, between September 12 and September 16, 2024.
- **FLOOD · Opole, Poland · 2024-09-12**
  - During Storm Boris in mid-September 2024, the Oder river at Opole, Poland rose to its highest level in over two decades.
- **FLOOD · ? · 2024-09-12**
  - Storm Boris caused flooding that spread across Poland, Czechia, Austria and Slovakia in mid-September 2024.
- **STORM · Vienna, Austria · 2024-09-12**
  - Storm Boris brought wind gusts of over 150 km/h to Vienna, Austria, in mid-September 2024.

## Key tool calls
- `search_gdacs_events(50.2758707, 17.1808923, 2024-09-12→2024-09-17)` → events[0]=Flood in Poland (Green)
- `get_historical_precipitation(50.2758707, 17.1808923, 2024-09-01→2024-09-17)` → total_mm=485.7, peak_daily_mm=171.0
- `get_river_discharge_glofas(50.2758707, 17.1808923, 2024-09-01→2024-09-17)` → peak_m3s=63.27, peak_date=2024-09-15, event_rank=1, years_compared=16
- `get_station_observations(50.2758707, 17.1808923, 2024-09-01→2024-09-17)` → no data (No station with daily data within 75 km for this range.)
- `get_european_river_gauges(50.2758707, 17.1808923, 2024-09-01→2024-09-17)` → no data (No national gauge with discharge data nearby. Gauge APIs are integrated for Fran)
- `search_gdacs_events(50.6668184, 17.9236408, 2024-09-10→2024-09-20)` → events[0]=Flood in Poland (Green)
- `find_nearest_water_body(50.6668184, 17.9236408, )` → {"tool": "find_nearest_water_body", "error": "HTTPStatusError: Server error '500 Internal Server Error' for url 'https://overpass.kumi.systems/api/interpreter?d
- `get_historical_precipitation(50.6668184, 17.9236408, 2024-08-29→2024-09-17)` → total_mm=132.5, peak_daily_mm=45.3
- `get_river_discharge_glofas(50.6668184, 17.9236408, 2024-08-29→2024-09-17)` → peak_m3s=825.04, peak_date=2024-09-16, event_rank=1, years_compared=21
- `get_european_river_gauges(50.6668184, 17.9236408, 2024-08-29→2024-09-17)` → no data (No national gauge with discharge data nearby. Gauge APIs are integrated for Fran)
- `search_gdacs_events(50, 17, 2024-09-10→2024-09-18)` → events[0]=Flood in Austria, Czech Republic, Germany, Poland, Romania, Slovakia (Orange)
- `find_nearest_water_body(50.43, 16.65, )` → {"tool": "find_nearest_water_body", "error": "HTTPStatusError: Server error '500 Internal Server Error' for url 'https://overpass.kumi.systems/api/interpreter?d
- `get_historical_precipitation(50.43, 16.65, 2024-08-29→2024-09-17)` → total_mm=206.7, peak_daily_mm=53.9
- `get_historical_precipitation(48.3, 15.6, 2024-08-29→2024-09-17)` → total_mm=276.5, peak_daily_mm=102.5
- `get_river_discharge_glofas(50.43, 16.65, 2024-08-29→2024-09-17)` → peak_m3s=84.7, peak_date=2024-09-17, event_rank=1, years_compared=11
- `get_river_discharge_glofas(48.3, 15.6, 2024-08-29→2024-09-17)` → peak_m3s=8490.8, peak_date=2024-09-16, event_rank=1, years_compared=11
- `get_station_observations(50.43, 16.65, 2024-08-29→2024-09-17)` → no data (No station with daily data within 75 km for this range.)
- `get_european_river_gauges(50.43, 16.65, 2024-08-29→2024-09-17)` → no data (No national gauge with discharge data nearby. Gauge APIs are integrated for Fran)
- `search_gdacs_events(48.2083537, 16.3725042, 2024-09-12→2024-09-16)` → events[0]=Flood in Slovakia (Green)
- `get_historical_winds(48.2083537, 16.3725042, 2024-09-11→2024-09-17)` → peak_speed_kmh=51.9, peak_gust_kmh=90.0
- `get_station_observations(48.2083537, 16.3725042, 2024-09-11→2024-09-17)` → max_total_precip_mm=420.1, max_daily_precip_mm=155.4, max_peak_gust_kmh=91.1, stations[0]=Wien / Hohe Warte (90.0)

## Final report

| Claim | Verdict | Rubric |
|---|---|---|
| Storm Boris brought more than 300 mm of rainfall to the town of Jeseník, Czechia, between September 12 and September 16, 2024. | SUPPORTS | 62% |
| During Storm Boris in mid-September 2024, the Oder river at Opole, Poland rose to its highest level in over two decades. | SUPPORTS | 75% |
| Storm Boris caused flooding that spread across Poland, Czechia, Austria and Slovakia in mid-September 2024. | SUPPORTS | 75% |
| Storm Boris brought wind gusts of over 150 km/h to Vienna, Austria, in mid-September 2024. | CONTRADICTS | 43% |

**Overall: CONTRADICTS · confidence 64%** · claude-cli:sonnet · 289 s · $1.43 API-equivalent · 16 LLM calls

## Rationales
- **SUPPORTS** — Storm Boris brought more than 300 mm of rainfall to the town of Jeseník, Czechia, between September 12 and September 16, 2024.: ERA5 precipitation at Jeseník (50.276N, 17.181E) for Sep 12-16, 2024 totals 445.9 mm (39.1+102.8+171.0+110.8+22.2), well over the claimed 300 mm, with a peak daily value of 171.0 mm on Sep 14 (overall peak across the full Sep 1-17 window was also 171.0 mm). This confirms extreme, flood-level rainfall. GDACS corroborates an Orange-level flood alert for 'Austria, Czech Republic, Germany, Poland, Romania, Slovakia' from 2024-09-12 to 2024-09-18, 49 km from the point, plus a nearby Green flood event in Poland 15 km away (Sep 13-15) — consistent with the widely-reported Storm Boris disaster. GloFAS river discharge at the nearest cell (50.275N, 17.225E) peaked at 63.27 m3/s on Sep 15, a 13.8x rise over the 4.57 m3/s baseline, ranked #1 (highest) among 16 years of annual maxima — strong independent confirmation of extreme flood conditions. No nearby Meteostat weather station (within 75 km) or European gauge (France/UK network only) had data, so those two tools are inconclusive/not applicable here, but the precipitation, GDACS, and GloFAS evidence together strongly support the claim.
- **SUPPORTS** — During Storm Boris in mid-September 2024, the Oder river at Opole, Poland rose to its highest level in over two decades.: Multiple independent lines of evidence confirm major, record-level flooding on the Oder at Opole during Storm Boris. GDACS recorded an Orange-level multi-country flood event ('Flood in Austria, Czech Republic, Germany, Poland, Romania, Slovakia') spanning 2024-09-12 to 2024-09-18, 113 km from Opole. ERA5 precipitation at the site shows 132.5 mm total over the 20-day window (2024-08-29 to 2024-09-17), with a sharp multi-day burst (18.9 mm on 9/12, 21.2 mm on 9/13, 45.3 mm peak on 9/14, 26.1 mm on 9/15) consistent with an extreme rain event hitting the region. Most decisively, GloFAS discharge at the river cell nearest Opole (50.625°N, 17.925°E) rose from a baseline of 92.5 m³/s to a peak of 825.04 m³/s on 2024-09-16 — an 8.9x rise over baseline — and this peak ranked #1 (highest) among 21 years of annual maxima compared, directly corroborating the claim of the highest level 'in over two decades.' European national gauge data (France/UK network) returned no coverage for Poland, and the OSM water-body lookup errored out, so those two tools provide no contradicting or additional evidence, but the GDACS, precipitation, and GloFAS discharge results together clearly support the claim.
- **SUPPORTS** — Storm Boris caused flooding that spread across Poland, Czechia, Austria and Slovakia in mid-September 2024.: GDACS recorded an Orange-alert flood event named 'Flood in Austria, Czech Republic, Germany, Poland, Romania, Slovakia' running 2024-09-12 to 2024-09-18, only 16 km from the query point, plus separate Green-alert entries for 'Flood in Poland' (Sep 13-15) and 'Flood in Slovakia' (Sep 14-16) — directly corroborating the claimed multi-country flood footprint. Precipitation (ERA5) near the Czech/Poland border (50.43N,16.65E) totaled 206.7 mm over the 20-day window with a peak of 53.9 mm on Sep 15; near Austria (48.3N,15.6E) totals reached 276.5 mm with a peak of 102.5 mm on Sep 14 — both extreme multi-day rainfall totals consistent with Boris's slow-moving rain bands. GloFAS discharge confirms flood-level river response in both countries: the Czech/Poland river cell (50.475N,16.675E) peaked at 84.7 m³/s on Sep 17, 4.3x its 60-day baseline (19.75 m³/s), ranked #1 (highest) of 11 years of annual maxima; the Austrian river cell (48.375N,15.625E) peaked at 8,490.8 m³/s on Sep 16, 3.4x baseline (2,516.7 m³/s), also ranked #1 of 11 years. These record-setting, multi-year-high discharge peaks in both countries, paired with the GDACS multi-country flood catalog entry and extreme rainfall totals, meet the bar for flood-level conditions. (find_nearest_water_body returned a server error with no data; get_station_observations and get_european_river_gauges found no station/gauge coverage near this remote point within range — these are inconclusive gaps, not contradicting evidence, and are outweighed by the GDACS and GloFAS findings.)
- **CONTRADICTS** — Storm Boris brought wind gusts of over 150 km/h to Vienna, Austria, in mid-September 2024.: GDACS confirms Storm Boris caused a major flood event (Orange alert) across Austria, Czech Republic, Germany, Poland, Romania and Slovakia from 2024-09-12 to 2024-09-18, corroborating the storm's occurrence and severity as a rain/flood event. However, the specific wind-gust claim of 'over 150 km/h' in Vienna is not supported by any wind data source. ERA5 reanalysis at Vienna's coordinates (48.21N, 16.37E) shows a peak gust of only 90.0 km/h on 2024-09-14, with sustained wind peaking at 51.9 km/h. Station observations corroborate this: Wien/Hohe Warte (4.7 km from Vienna center) measured a peak gust of exactly 90.0 km/h on 2024-09-14 (with heavy rainfall of 101.6 mm that day and 207.9 mm total over the week), and the more distant St. Pölten Landhaus station (55.1 km away) recorded a slightly higher peak gust of 91.1 km/h with 420.1 mm total precipitation. The maximum measured or modeled gust anywhere near Vienna (91.1 km/h) is roughly 59 km/h below the claimed 150 km/h threshold -- a difference far too large to attribute to normal measurement/siting variance or generous interpretation of claim phrasing. While Storm Boris was indeed a significant and damaging flood-producing storm, the specific wind gust magnitude claimed (over 150 km/h) in Vienna is contradicted by both reanalysis and station-measured data.
