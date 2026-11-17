"""Inflation analysis linked to monthly coconut supply observations."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from .common import table, chart, download_table


def _monthly_panel(wb):
    d = wb.monthly.copy()
    if 'inflation_pct' not in d or d.inflation_pct.notna().sum() == 0:
        return pd.DataFrame()

    # Inflation is an area-wide monthly indicator. If repeated on each
    # barangay row, retain one monthly value rather than summing duplicates.
    inflation = (d.dropna(subset=['inflation_pct'])
                   .groupby('date', as_index=False)
                   .agg(inflation_pct=('inflation_pct', 'median'),
                        inflation_records=('inflation_pct', 'count'),
                        inflation_min=('inflation_pct', 'min'),
                        inflation_max=('inflation_pct', 'max')))
    inflation['inflation_variation'] = inflation.inflation_max - inflation.inflation_min

    supply = (d.groupby('date', as_index=False)
                .agg(coconut_supply_kg=('harvest_kg', lambda s: s.sum(min_count=1)),
                     harvested_area_ha=('area_ha', 'sum'),
                     barangays_with_harvest=('harvest_kg', 'count')))
    supply['year'] = supply.date.dt.year
    supply['month'] = supply.date.dt.month
    panel = inflation.merge(supply, on='date', how='left').sort_values('date')
    panel['year'] = panel.date.dt.year
    panel['month'] = panel.date.dt.month
    return panel


def render(wb):
    st.title('Coconut Supply and Local Inflation Analysis')
    st.write('Examine historical inflation and its observed relationship with recorded coconut supply in Rosario.')
    panel = _monthly_panel(wb)
    if panel.empty:
        st.warning('No usable Inflation Rate (%) values were found. Upload a dataset with verified monthly inflation observations to use this module.')
        st.caption('The existing coconut production analysis remains available. Inflation values are not generated or assumed by this module.')
        return

    valid = panel.dropna(subset=['inflation_pct']).copy()
    first, last = valid.date.min(), valid.date.max()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Monthly inflation observations', f'{len(valid):,}')
    c2.metric('Average recorded inflation', f'{valid.inflation_pct.mean():.2f}%')
    c3.metric('Latest recorded inflation', f'{valid.sort_values("date").iloc[-1].inflation_pct:.2f}%')
    c4.metric('Period covered', f'{first:%Y}–{last:%Y}')
    st.caption('Inflation values are treated as monthly area-wide observations. When the same monthly rate appears on multiple barangay rows, the module uses the monthly median to avoid counting repeated values multiple times.')

    st.subheader('Historical inflation trend')
    trend = valid[['date', 'inflation_pct']].sort_values('date')
    chart(alt.Chart(trend).mark_line(point=True).encode(
        x=alt.X('date:T', title='Month'),
        y=alt.Y('inflation_pct:Q', title='Inflation rate (%)', scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip('date:T', title='Month'), alt.Tooltip('inflation_pct:Q', title='Inflation (%)', format='.2f')]
    ).properties(height=320).interactive())

    annual = (valid.groupby('year', as_index=False)
              .agg(average_inflation_pct=('inflation_pct', 'mean'),
                   minimum_inflation_pct=('inflation_pct', 'min'),
                   maximum_inflation_pct=('inflation_pct', 'max'),
                   months_recorded=('inflation_pct', 'count')))
    st.subheader('Yearly inflation summary')
    chart(alt.Chart(annual).mark_bar().encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('average_inflation_pct:Q', title='Average inflation (%)'),
        tooltip=['year', 'average_inflation_pct', 'minimum_inflation_pct', 'maximum_inflation_pct', 'months_recorded']
    ).properties(height=280))
    table(annual.round(3))
    download_table(annual, 'yearly_inflation_summary.csv')

    st.subheader('Coconut supply and inflation')
    paired = panel.dropna(subset=['inflation_pct', 'coconut_supply_kg']).copy()
    paired = paired[paired.coconut_supply_kg.notna()]
    left, right = st.columns(2)
    with left:
        chart(alt.Chart(paired).mark_circle(size=75, opacity=.65).encode(
            x=alt.X('coconut_supply_kg:Q', title='Recorded coconut harvest (kg)'),
            y=alt.Y('inflation_pct:Q', title='Inflation rate (%)'),
            tooltip=[alt.Tooltip('date:T', title='Month'), 'coconut_supply_kg', 'inflation_pct']
        ).properties(height=300).interactive())
    with right:
        st.write('Monthly supply and inflation observations')
        table(paired[['date', 'coconut_supply_kg', 'inflation_pct', 'inflation_records']].tail(18).round(3))

    if len(paired) >= 3 and paired.coconut_supply_kg.nunique() > 1 and paired.inflation_pct.nunique() > 1:
        pearson = paired[['coconut_supply_kg', 'inflation_pct']].corr(method='pearson').iloc[0, 1]
        spearman = paired[['coconut_supply_kg', 'inflation_pct']].corr(method='spearman').iloc[0, 1]
        a, b = st.columns(2)
        a.metric('Pearson correlation', f'{pearson:.3f}')
        b.metric('Spearman correlation', f'{spearman:.3f}')
        st.caption('Correlation describes association in the available paired months. It does not establish that coconut supply causes inflation. Other economic conditions may also influence inflation.')
    else:
        st.info('At least three paired monthly observations with variation in both variables are needed to calculate correlation.')

    st.subheader('Lagged supply relationship')
    lagged = paired.sort_values('date').copy()
    lagged['supply_previous_month_kg'] = lagged.coconut_supply_kg.shift(1)
    lag_rows = []
    for lag, label in [(0, 'Same month'), (1, 'Supply 1 month earlier'), (2, 'Supply 2 months earlier'), (3, 'Supply 3 months earlier')]:
        col = 'coconut_supply_kg' if lag == 0 else f'supply_lag_{lag}'
        lagged[col] = lagged.coconut_supply_kg.shift(lag)
        subset = lagged[[col, 'inflation_pct']].dropna()
        corr = subset.corr().iloc[0, 1] if len(subset) >= 3 and subset[col].nunique() > 1 and subset.inflation_pct.nunique() > 1 else np.nan
        lag_rows.append({'Comparison': label, 'Paired months': len(subset), 'Correlation': corr})
    table(pd.DataFrame(lag_rows).round(3))
    st.caption('Lag comparisons are exploratory. They do not prove a delayed causal effect, and fewer observations are available at longer lags.')

    with st.expander('Inflation data quality and monthly records'):
        st.write('Check whether inflation was repeated consistently across barangay rows for each month.')
        quality = valid[['date', 'inflation_records', 'inflation_min', 'inflation_max', 'inflation_variation']].copy()
        table(quality.round(4))
        inconsistent = quality[quality.inflation_variation.abs() > 1e-9]
        if not inconsistent.empty:
            st.warning(f'{len(inconsistent)} months contain different inflation values across source rows. Verify the source and decide which geographic level the inflation rate represents.')
        else:
            st.success('No within-month differences were detected among the available repeated inflation values.')
        download_table(panel, 'monthly_coconut_supply_inflation.csv')
