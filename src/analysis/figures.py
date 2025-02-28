import matplotlib.pyplot as plt
import polars as pl
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

    draw_pie_chart(dp_per_dataset, 'source_dataset', 'source_datasets', number_of_dp)
    draw_pie_chart(dp_per_domain, 'domain', 'domains', number_of_dp)
    draw_pie_chart(dp_per_task, 'task', 'tasks', number_of_dp)

    # draw non-N/A domains
    dp_per_domain = dp_per_domain.filter(dp_per_domain['domain'] != 'N/A')
    draw_pie_chart(dp_per_domain, 'domain', 'domains_non_na', number_of_dp)

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

def draw_pie_chart(data: pl.DataFrame, label_col, figname, total_dp):
    os.makedirs("output/figures/basic_stats", exist_ok=True)

    # increase fontsize
    plt.rcParams.update({'font.size': 12})

    plt.figure(figsize=(10, 7))
    plt.text(-1.2, 1.2, f"Total Datapoints: {total_dp}", fontsize=12)
    plt.pie(data['count'], labels=data[label_col], autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'output/figures/basic_stats/{figname}.png')
    plt.close()

def draw_3col_pie_chart(data: pl.DataFrame):
    # Ensure the output directory exists
    os.makedirs("output/figures/basic_stats", exist_ok=True)
    
    # Set the theme and font size for the plot

    plt.rcParams.update({'font.size': 12})

    # Create the pie chart
    plt.figure(figsize=(10, 7))
    plt.pie(data['count'], labels=data['source_dataset'], colors=data['color'], autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    
    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'output/figures/basic_stats/null_nonnull_pie_chart.png')
    plt.close()

def plot_histogram(data: pl.DataFrame, bin_count=10):
    os.makedirs("output/figures/basic_stats", exist_ok=True)
    hist = data['similarities'].hist(bin_count=bin_count)

    plt.figure(figsize=(10, 7))
    plt.hist(data, bins=50, alpha=0.7, color='b')
    plt.xlabel('Length of Context')
    plt.ylabel('Frequency')
    plt.title('Histogram of Context Length')
    plt.savefig('output/figures/basic_stats/histogram.png')
    plt.close()

def plot_pie(data: dict, figname, output_dir):
    """ Data: dict of {subfigure: (data_occurences, labels)}
        Figname: name of the figure
    """
    folder = f"{output_dir}"
    os.makedirs(folder, exist_ok=True)
    fig_count = len(data.keys())
    # Dynamically determine the number of rows and columns
    cols = int(np.ceil(np.sqrt(fig_count)))  # Closest square layout
    rows = int(np.ceil(fig_count / cols))    # Adjust rows accordingly
    fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), squeeze=True)
    if fig_count > 1:
        axs = axs.ravel()
    else:
        axs = [axs]

    for idx, key in enumerate(data.keys()):
        _data = data[key]
        axs[idx].pie(data[key][1], labels=data[key][0], autopct='%1.1f%%', startangle=140)
        axs[idx].axis('equal')
        axs[idx].set_title(f"{key}; n={sum(data[key][1])}")
    
    fig.suptitle(figname, fontsize=20)
    fig.savefig(f'{folder}/{figname}_pie.png')
    plt.close()