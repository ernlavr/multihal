import polars as pl
# import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np



def plot_ds_stats(df: pl.DataFrame):
    dataset_name = df['source_dataset'].unique().to_list()
    number_of_dp = df.shape[0]
    dp_per_dataset = df.group_by('source_dataset').count().sort('source_dataset')
    dp_per_domain = df.group_by('domain').count().sort('domain')
    dp_per_task = df.group_by('task').count().sort('task')
    # datapoints per context where context isnt null
    dp_with_context = df['context'].filter(df['context'].is_not_null()).count()

    draw_pie_chart(dp_per_dataset, 'source_dataset', number_of_dp)
    draw_pie_chart(dp_per_domain, 'domain', number_of_dp)
    draw_pie_chart(dp_per_task, 'task', number_of_dp)

    # get nulls/non_nulls
    nulls = df['context'].is_null()
    non_nulls = df['context'].is_not_null()
    
    nulls_df = df.filter(nulls)['source_dataset'].value_counts()
    nulls_df = nulls_df.with_columns(
        color=pl.lit('red')
    )
    non_nulls_df = df.filter(non_nulls)['source_dataset'].value_counts()
    non_nulls_df = non_nulls_df.with_columns(
        color=pl.lit('green')
    )
    nulls = pl.concat([nulls_df, non_nulls_df], how='diagonal')
    
    draw_3col_pie_chart(nulls)        

    # self.draw_pie_chart(dp_with_context, 'context', number_of_dp)

def draw_pie_chart(data: pl.DataFrame, figname, total_dp):
    os.makedirs("output/figures", exist_ok=True)

    # increase fontsize
    plt.rcParams.update({'font.size': 12})

    plt.figure(figsize=(10, 7))
    plt.text(-1.2, 1.2, f"Total Datapoints: {total_dp}", fontsize=12)
    plt.pie(data['count'], labels=data[figname], autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'output/figures/{figname}.png')
    plt.close()

def draw_3col_pie_chart(data: pl.DataFrame):
    # Ensure the output directory exists
    os.makedirs("output/figures", exist_ok=True)
    
    # Set the theme and font size for the plot

    plt.rcParams.update({'font.size': 12})

    # Create the pie chart
    plt.figure(figsize=(10, 7))
    plt.pie(data['count'], labels=data['source_dataset'], colors=data['color'], autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    
    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'output/figures/null_nonnull_pie_chart.png')
    plt.close()

def plot_histogram(data: pl.DataFrame, bin_count=10):
    os.makedirs("output/figures", exist_ok=True)
    hist = data['similarities'].hist(bin_count=bin_count)

    plt.figure(figsize=(10, 7))
    plt.hist(data, bins=50, alpha=0.7, color='b')
    plt.xlabel('Length of Context')
    plt.ylabel('Frequency')
    plt.title('Histogram of Context Length')
    plt.savefig('output/figures/histogram.png')
    plt.close()