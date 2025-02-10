    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr
    import statsmodels.api as sm
    # Load the data sets
    tourism_df = pd.read_csv('/content/drive/MyDrive/tourism_data (1).csv')
    etourism_df = pd.read_csv('/content/drive/MyDrive/etourism_data (1).csv
    ')
    # Merge the data sets by relevant columns
    merged_df = pd.merge(tourism_df, etourism_df, on='year')
    # Clean and preprocess the data
    merged_df = merged_df.dropna() # remove missing values
    merged_df['international_tourists'] = merged_df['international_tourists
    '].astype(int)
    merged_df['domestic_tourists'] = merged_df['domestic_tourists'].astype(
    int)
    merged_df['online_bookings'] = merged_df['online_bookings'].astype(int)
    # Basic statistics
    print(merged_df.describe())
    # Scatter plot of international tourists and online bookings
    sns.scatterplot(x='international_tourists', y='online_bookings', data=m
    erged_df)
    plt.show()
    # Line plot of domestic tourists and online bookings over time
    sns.lineplot(x='year', y='domestic_tourists', data=merged_df, label='Do
    mestic tourists')
    sns.lineplot(x='year', y='online_bookings', data=merged_df, label='Onli
    ne bookings')
    plt.legend()
    plt.show()
    # Compute correlation between online bookings and international tourist
    s
    corr, _ = pearsonr(merged_df['online_bookings'], merged_df['internation
    al_tourists'])
    print(f'Pearson correlation coefficient: {corr:.2f}')
    # Compute correlation between online bookings and domestic tourists
    corr, _ = pearsonr(merged_df['online_bookings'], merged_df['domestic_to
    urists'])
    print(f'Pearson correlation coefficient: {corr:.2f}')
    # Load the data sets
    tourism_df = pd.read_csv('/content/drive/MyDrive/tourism_data (1).csv')
    etourism_df = pd.read_csv('/content/drive/MyDrive/etourism_data (1).csv
    ')
    # Merge the data sets by relevant columns
    merged_df = pd.merge(tourism_df, etourism_df, on='year')
    # Clean and preprocess the data
    merged_df = merged_df.dropna() # remove missing values
    merged_df['international_tourists'] = merged_df['international_tourists
    '].astype(int)
    merged_df['domestic_tourists'] = merged_df['domestic_tourists'].astype(
    int)
    merged_df['online_bookings'] = merged_df['online_bookings'].astype(int)
    # Compute correlation between online bookings and international tourist
    s
    corr, _ = pearsonr(merged_df['online_bookings'], merged_df['internation
    al_tourists'])
    print(f'Pearson correlation coefficient (online bookings vs internation
    al tourists): {corr:.2f}')
    # Compute correlation between online bookings and domestic tourists
    corr, _ = pearsonr(merged_df['online_bookings'], merged_df['domestic_to
    urists'])
    print(f'Pearson correlation coefficient (online bookings vs domestic to
    urists): {corr:.2f}')
    # Build a linear regression model with domestic tourists as the indepen
    dent variable and international tourists and online bookings as the dep
    endent variables
    X = merged_df['domestic_tourists']
    y = merged_df[['international_tourists', 'online_bookings']]
    y = sm.add_constant(y) # Add a constant term to the model
    model = sm.OLS(X, y).fit()
    # Print the summary of the regression model
    print(model.summary())
    # Plot the linear regression line
    sns.lmplot(x='domestic_tourists', y='online_bookings', data=merged_df,
    scatter_kws={'alpha':0.3})
    plt.xlabel('Domestic tourists')
    plt.ylabel('Online bookings')
    plt.title('Linear regression with Domestic tourists as independent vari
    able')
    plt.show()
    # Plot the residual plot
    sns.residplot(x='domestic_tourists', y='online_bookings', data=merged_d
    f, scatter_kws={'alpha':0.3})
    plt.xlabel('Domestic tourists')
    plt.ylabel('Residuals')
    plt.title('Residual plot with Domestic tourists as independent variable
    ')
    plt.show()