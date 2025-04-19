import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(layout="wide", 
    page_title="NFL GM Analysis",
    page_icon="🧊",
    menu_items={
        'About': "Check out my [Portfolio](https://sites.google.com/view/seth-lanza-portfolio/home), Get in touch at Seth.Lanza@gmail.com, Connect on [Linkedin](https://www.linkedin.com/in/sethlanza/), Check my [Twitter](https://x.com/SethDataScience), or Check out my [Github](https://github.com/SethTheDataScientist?tab=repositories)"
    })


st.title("NFL Draft Analysis")
# Sidebar for main sections
main_menu = st.sidebar.radio(
    "Select an Analysis",
    ["GM Analysis", "Player Predictions"]
)
if main_menu == "GM Analysis":
    st.header("GM Analysis")
    st.session_state.target_column = None
    st.write("Here is my model output from an analysis of NFL GMs. The data used for training is the draft picks from 2003 to 2024 drafts along with features about the prospects including athletic testing, consensus big board rank, and NGS scores. The model was trained to predict the class label of the GM most likely to draft that player. The model was trained with recency weighting by an exponential function centered on the 2021 draft with an alpha of 0.5 (2024 = 4.48, 2021 = 1.0, and 2003 = 0.00012). Given all the biases inherent with the draft and any post-hoc analysis, the model is not very accurate but it is at least interesting.")
    st.write("The model was trained using a XGBoost Classifier with the following hyperparameters and model performance metrics:")
    st.write("Hyperparameters: 'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8")
    st.write("Model Performance: log_loss: 4.2819, roc_auc_ovr_macro: 0.3894, top_1_accuracy: 0.0279, top_5_accuracy: 0.1540")
    st.image("gm_data/feature_importance.png", caption="XGBoost Feature Importances")


    # Read in the upcoming draft table
    st.session_state.current_prospects = pd.read_csv("gm_data/current_year.csv")
    st.subheader("Current Year Predictions")
    st.write("Let's jump right to the good stuff and see the model output for the upcoming draft class. Below is the top 5 predictions and their probabilites for players in this upcoming draft sorted by most likely first label.")
    st.dataframe(st.session_state.current_prospects, use_container_width=True)

    st.subheader("Similarity Plot")
    # Read in the PCA means table
    st.session_state.pca_means = pd.read_csv("gm_data/GM_PCA.csv")

    # Select target column as the GM label
    st.session_state.target_column = st.selectbox("Select the specific GM from either a team's most recent GM to make a draft pick or if that GM picked at least 50 draft picks.", st.session_state.pca_means.label.unique())

    st.write("The similarity plot is derived by the principal components of the average features for each GM. The closer the GMs are in the plot, the more similar their average features are.")
    st.write("The actual values are not relevant here, just the relative distances and locations near each other. The first two principal components explain 70% of the variance in the data.")

    # Filter the PCA means table to get the selected GM
    selected = st.session_state.pca_means[st.session_state.pca_means['label'] == st.session_state.target_column]

    pca_means = st.session_state.pca_means[st.session_state.pca_means['label'] != st.session_state.target_column]


    # Plot button
    if st.button("Similarity Plot"):
        plt.figure(figsize=(10, 7))

        # Create scatterplot
        sns.scatterplot(data=pca_means, x='PC1', y='PC2', hue='label', palette='Set2', s=70, alpha=0.8)

        # Add jittered text annotations for each point
        for i in range(pca_means.shape[0]):
            jitter_x = np.random.uniform(-1, 1)  # Jitter in x direction
            jitter_y = np.random.uniform(-1, 1)  # Jitter in y direction
            
            plt.text(pca_means['PC1'].iloc[i] + jitter_x, 
                    pca_means['PC2'].iloc[i] + jitter_y, 
                    pca_means['label'].iloc[i], 
                    fontsize=9, alpha=0.7, color='black', ha='center', va='center')

        # Title and labels
        plt.title("PCA - First Two Components")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        # Put dashed lines at every 5 units for both axes
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)


        sns.scatterplot(data=selected, x='PC1', y='PC2', s=100, color = 'red', edgecolor = 'black')
        plt.text(selected['PC1'], 
                    selected['PC2'] - 2, 
                    selected['label'].values[0],
                    fontsize=10, alpha=1, color='black', ha='center', va='center')


        # Remove the legend
        plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left').set_visible(False)

        # Tight layout to ensure everything fits
        plt.tight_layout()

        # Show the plot
        st.pyplot(plt)


    # Read in the SHAP values table
    st.subheader("SHAP Analysis")
    st.write("Here are some of the SHAP values from the analysis. SHAP values show an effective way of explaining how the model has learned to create the output. This is for the selected GM to give a sense of what the model found them to draft more consistently.")
    with open("gm_data/shap_full.pkl", "rb") as f:
        st.session_state.shap_values = pickle.load(f)
    st.session_state.categories = pd.read_csv("gm_data/categories.csv")['0']
    st.session_state.categories = pd.Index(st.session_state.categories)
    st.session_state.test_X = pd.read_csv("gm_data/test_X.csv")
    st.session_state.class_idx = st.session_state.categories.get_loc(st.session_state.target_column) 
    # If they don't match, you can force alignment:
    if list(st.session_state.shap_values.feature_names) != st.session_state.test_X.columns.tolist():
        # Reorder test_X to match SHAP feature names
        st.session_state.test_X = st.session_state.test_X[list(st.session_state.shap_values.feature_names)]
    # Extract SHAP values for that class
    st.session_state.shap_class_values = st.session_state.shap_values.values[:, :, st.session_state.class_idx]

    # Then create a new SHAP Explanation object for that class
    shap_class = shap.Explanation(
        values=st.session_state.shap_class_values,
        base_values=st.session_state.shap_values.base_values[:, st.session_state.class_idx],
        data=st.session_state.shap_values.data,
        feature_names=st.session_state.test_X.columns
    )

    st.subheader("SHAP Beeswarm Plot")
    fig, ax = plt.subplots(figsize=(10, 8))  # Create figure with explicit size
    shap.plots.beeswarm(shap_class, max_display=20, show=False)  # Add show=False
    plt.tight_layout()  # Ensure good spacing
    st.pyplot(fig)  # This should now work

    st.subheader("SHAP Bar Plot")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_class, max_display=20, show=False)
    plt.tight_layout()
    st.pyplot(fig2)

    st.subheader("Waterfall for First Prediction in dataset")
    st.write("This shows how the model arrived at the output for that player. The features are sorted by their SHAP value and the red bars are pushing the prediction higher while the blue bars are pushing it lower. The base value is the average model output for all players in the dataset.")
    fig3, ax3 = plt.subplots()
    shap.plots.waterfall(shap_class[0])
    st.pyplot(fig3)

    st.session_state.shap_column = st.selectbox("Select the specific feature to see how the SHAP values change across that feature for this specific GM", set(shap_class.feature_names))

    feature_index = list(shap_class.feature_names).index(st.session_state.shap_column)
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    shap.dependence_plot(
        feature_index, 
        shap_class.values,
        shap_class.data,
        feature_names=shap_class.feature_names,
        show=False,
        ax=ax4
    )
    plt.tight_layout()
    st.pyplot(fig4)


    # Read in the averages table
    st.session_state.averages = pd.read_csv("gm_data/averages.csv")
    st.subheader("Other Tables")
    st.write("Below are some tables of outputs of the model by GM or prospect.")
    st.write("This first table is the averages of the features for each GM for all players drafted by that GM.")

    st.dataframe(st.session_state.averages, use_container_width=True)


    # Read in the players and their top 5 GMs
    st.session_state.top_5_labels = pd.read_csv("gm_data/top_5_labels.csv")
    st.write("Next is a list of all the players in the sample by the five most likely GMs to draft them.")

    st.dataframe(st.session_state.top_5_labels, use_container_width=True)

    # Read in the players by if they had a specific GM within their top 5

    st.session_state.all_players_top5 = pd.read_csv("gm_data/all_players_top5.csv")

    st.write("Finally is a list of the players by if they had a GM within their top 5 most likely outcomes.")

    st.dataframe(st.session_state.all_players_top5, use_container_width=True)

elif main_menu == 'Player Predictions':
    tab1, tab2 = st.tabs(['WR Model', 'RB Model'])
    with tab1:
        st.subheader("WR Model")

        st.write("This tab is for a model that I created to predict NFL success based on college data for Wide Receivers. I will go into all the details about the model later, but let's jump to the good stuff with this year's predictions. (I don't filter to draftable prospects, this is anyone who played WR in 2024 with enough snaps to qualify.)")
        # Read in the upcoming draft table      
        
        # Create an expandable section for the data dictionary
        with st.expander("📊 Column Descriptions", expanded=False):
            # Create two columns
            col1, col2 = st.columns(2)
            
            # First column with descriptions
            with col1:
                st.markdown("**Prediction & Value Metrics:**")
                st.markdown("• **predicted_value**: Output of the model")
                st.markdown("• **min/mean/max values**: Results from stepping combine values up/down and running the model with new athleticism numbers")
                st.markdown("• **Value**: Composite metric created using multiple other features")
                st.markdown("• **WAR**: PFF wins-above-replacement clone based on the PFF grade")
                st.markdown("• **athleticism score**: Score from NextGen Stats")
                
                st.markdown("**Performance Metrics:**")
                st.markdown("• **RR**: Percentile of routes run")
                st.markdown("• **TPRR**: Percentile of targets per route run")
                st.markdown("• **YPRR**: Percentile of yards per route run")
                st.markdown("• **TDPRR**: Percentile of touchdowns per route run")
                st.markdown("• **ADOT**: Percentile of average depth of target")
                st.markdown("• **YAC**: Percentile of yards after catch per target")
                st.markdown("• **ContestedTile**: Percentile of contested target rate")
                st.markdown("• **Slot rate**: Percentage of routes run from the slot")
            
            # Second column with descriptions
            with col2:
                st.markdown("**Player Classification:**")
                st.markdown("• **NonSeparator**: Value based on if average depth of target is high enough and contested target rate is also too high, signifying lack of separation ability")
                st.markdown("• **Filter NonSeparator**: If non-separator percentage over 50% for career")
                st.markdown("• **Filter Solid**: If player wasn't a non-separator or a gadget player")
                st.markdown("• **Filter Gadget**: If ADOT percentile was below 20%")
                
                st.markdown("**Background Info:**")
                st.markdown("• **Last_Season**: The last year with data in college")
                st.markdown("• **Seasons**: Number of seasons of data available")
                st.markdown("• **Strength Power 5**: If the last year is from a school in the formerly Power 5")
                
                st.markdown("**Combine/Athletic Metrics:**")
                st.markdown("• **Combine metrics**: Various standard combine measurement metrics (height, weight, speed, etc.)")

        st.session_state.wr_current_prospects = pd.read_csv("wr_data/current_year_wr.csv")
        st.subheader("Current Year Predictions")
        st.dataframe(st.session_state.wr_current_prospects, use_container_width=True)

        st.subheader("Specific Player Distribution")
        st.session_state.specific_wr = st.selectbox("Select a player to see their distribution of their min, max, and predicted values. These are generated from the stepping around their actual combine numbers. If a player's prediction is near their max it means that they wouldn't be considered any better if they were a better athlete than they tested at the combine. If they are near their min, it means that the projections wouldn't get much worse if they were a worse athlete than they tested at the combine.", st.session_state.wr_current_prospects.ID.unique())

        plot_df = st.session_state.wr_current_prospects[st.session_state.wr_current_prospects.ID == st.session_state.specific_wr]
        row = plot_df.copy()
        # Parameters
        mean = row['predicted_value'].item()
        min_val =  row['min_value'].item()
        max_val =  row['max_value'].item()
        player = st.session_state.specific_wr

        distance_to_min = mean - min_val
        distance_to_max = max_val - mean
        max_distance = max(distance_to_min, distance_to_max)
        std_dev_estimate2 = max_distance / 3  # Assuming max distance is 3 standard deviations

        # Choose appropriate std_dev estimation method
        std_dev = std_dev_estimate2  # Change to std_dev_estimate2 if needed

        # Generate x values for plotting
        x = np.linspace(min_val - std_dev, max_val + std_dev, 1000)

        # Calculate normal PDF
        y = stats.norm.pdf(x, mean, std_dev)

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot the normal distribution
        plt.plot(x, y, 'r-', lw=2, label='Normal Distribution')

        # Shade the area between min and max
        plt.fill_between(x, y, where=(x >= min_val) & (x <= max_val), 
                        color='skyblue', alpha=0.4, label='Min-Max Range')

        # Add vertical lines for min, max, and mean
        plt.axvline(min_val, color='green', linestyle='--', alpha=0.7, label=f'Minimum = {min_val}')
        plt.axvline(max_val, color='purple', linestyle='--', alpha=0.7, label=f'Maximum = {max_val}')
        plt.axvline(mean, color='black', linestyle='-', alpha=0.7, label=f'Mean = {mean}')

        # Add labels and title
        plt.xlabel('Predicted_Output')
        plt.ylabel('Probability Density')
        plt.title(f'{player} : Normal Distribution with Min={min_val:.2f}, Mean={mean:.2f}, Max={max_val:.2f}, and Estimated Std Dev={std_dev:.2f}')
        plt.grid(True, alpha=0.3)


        plt.xlim(0,100)
        # Show the plot
        plt.tight_layout()
        # Show the plot
        st.pyplot(plt)

        
        # Read in the upcoming draft table
        st.session_state.wr_full_prospects = pd.read_csv("wr_data/full_list_prospects_wr.csv")
        st.subheader("Full Predictions")
        st.write('Here is the full table for all players in the dataset along with their NFL target variable to see the biggest misses in both directions. I will touch more on the specific misses below the table.')
        st.dataframe(st.session_state.wr_full_prospects, use_container_width=True)
        st.write('The biggest underpredictions were Ladd McConkey, Amon-Ra St. Brown, Devin Funchess, Zay Flowers, and Xavier Legette. Also notable names in the top 10 are Brandon Aiyuk, Jordan Addison, and Terry McLaurin.')
        st.write("It is worth noting that some of those players are rookies and might not maintain their current level of play. As they say, you're never wrong, just early. I will also say that a lot of the misses are for guys who are undersized as the model doesn't seem to like that a lot, and you can differentiate the underpredictions who became good by their max value being higher than the rest of them showing the upside was there at least. Admittedly, I'm not sure this model would ever pick up on a guy like St. Brown.")

        st.write('On the flip side of the coin, biggest overpredictions of note were Andy Isabella, Skyy Moore, Danny Gray, Parris Campbell, and Dazz Newsome')
        st.write("A lot of these player are also a bit undersized and for some player, most notably Parris Campbell, he falls into the Gadget filter that is when your ADOT is too low. I don't know why the model wasn't more punative with that filter as I initially definied it in my older models by scrolling until I found someone who I felt was at least alright in the NFL. They other misses were all decently early picks in the draft who showed some promise when you factor in some PFF features, but just never panned out in the pros.")


        st.subheader("Model Process and Explanations")
        st.write("Below I am going to talk out the different choices I made during the modeling process and why. Then I'll show some of the model evaluation with SHAP plots, correlation plots, and error metrics.")
        st.write("The first choice that I made was to determine what I was going to use as the target variable since whatever you tell the model to predict, it will. The target variable is therefore a very important decision. I chose to create a composite metric that is a weighted average of a handful of features that I use and have found to be very indicative of NFL success. These featres are Routes Run, Targets per route run, yards per route run, touchdowns per route run, average depth of target, and yards after catch. These values are converted to percentiles and scaled according to the following weights: RR = 10, TPRR = 4, YPRR = 4, TDPRR = 3, ADOT = 3, YAC = 1. I found that to be a good mix of volume and efficiency metrics and guys like Justin Jefferson and Calvin Johnson dominate the top of the list.")
        st.write("I also filtered this to only include a player's first 4 seasons in the NFL to try and capture as many players as possible and avoid aging curves and survivorship bias. I adjusted down the value of a first season for a player to be 50% weighted toward the average first year player. I then created a scaling feature using the logit function to weight the model more towards predicting the top end wide receivers correctly, while putting less emphasis on the lower value players.")
        st.write("I created the college data using the values as described above and I created features for best and worst features for an individual season to capture if someone had a breakout year that wasn't captured in the raw average. I also imputed the combine data for players that didn't have data listed by using their production scores with a KNN to gather their athleticism data. I found this to boost the model quite a bit for individual projections.")
        st.write("Then I removed players who weren't drafted to ensure that the model wasn't training on guys who played in college but weren't legit prospects. I then created monotonic constraints for the features so the model learned that being faster is better for example and not try to find pockets of 40 yard dash times with higher value. I don't want the model to learn that running a 4.45 is better than a 4.37 just because some guys who ran a 4.37 weren't good in the pros.")
       

        st.subheader("Model Evaluation")
        st.write("I trained an XGBoost model with the following hyperparameters [learning_rate: 0.1, max_depth: 7, num_estimators: 100, col_subsample: 0.8, subsample: 0.8].")
        st.write("The model performed pretty well on the test set with the following metrics [R^2: 0.384, rmse: 0.177, mae: 0.150]. Below is the correlation plot between the predicted and actual target for the entire sample outside of the 2024 prospects.")
        st.image("wr_data/correlation_scatter_plot.png", caption="Scatter Plot of Predicted vs Actual with line of best fit.")

        st.write("I also created a SHAP analysis of the model with the beeswarm and bar plots showing the overall model features and their importance and the waterfall plot showing an example with the first row in the sample to show how the prediction is made.")
        st.image("wr_data/shap_bar_plot.png", caption="Bar Plot of shap values for the features in the model")
        st.image("wr_data/shap_beeswarm_plot.png", caption="Beeswarm plot for shap values and the corresponding feature values")
        st.image("wr_data/shap_waterfall_plot.png", caption="Waterfall plot showing the model prediction for the first player in the sample")


        st.session_state.wr_shap_column = st.selectbox("Select the specific feature to see how the SHAP values change across that feature", sorted(set(st.session_state.wr_full_prospects.columns)))
        st.image(f"wr_data/shap_scatter_plot_{st.session_state.wr_shap_column}.png", caption="Shap Scatter plot showing the shap values for a specific feature and the more correlated secondary feature")

    with tab2:
        st.subheader("RB Model")

        st.write("This tab is for a model that I created to predict NFL success based on college data for Running Backs. I will go into all the details about the model later, but let's jump to the good stuff with this year's predictions. (I don't filter to draftable prospects, this is anyone who played RB in 2024 with enough snaps to qualify.)")
        # Read in the upcoming draft table      
        
        # Create an expandable section for the data dictionary
        with st.expander("📊 Column Descriptions", expanded=False):
            # Create two columns
            col1, col2 = st.columns(2)
            
            # First column with descriptions
            with col1:
                st.markdown("**Prediction & Value Metrics:**")
                st.markdown("• **predicted_value**: Output of the model")
                st.markdown("• **min/mean/max values**: Results from stepping combine values up/down and running the model with new athleticism numbers")
                st.markdown("• **RecWAR**: PFF wins-above-replacement clone based on the receiving grade")
                st.markdown("• **RushWAR**: PFF wins-above-replacement clone based on the rushing grade")
                st.markdown("• **WAR**: PFF wins-above-replacement clone based on sum of component grades")
                
                st.markdown("**Performance Metrics:**")
                st.markdown("• **Attempts**: Percentile of total rushing attempts")
                st.markdown("• **ForcedMissedTackleRate**: Percentile of forced missed tackle rate from PFF")
                st.markdown("• **ExplosiveRate**: Percentile of explosive rushing rate (10+ yard rushes)")
                st.markdown("• **TDP**: Percentile of touchdowns per carry")
                st.markdown("• **YardsAfterContact**: Percentile of yards after contact per attempt")
                st.markdown("• **YPRR**: Percentile of yards per route run")
            
            # Second column with descriptions
            with col2:                
                st.markdown("**Background Info:**")
                st.markdown("• **Last_Season**: The last year with data in college")
                st.markdown("• **Seasons**: Number of seasons of data available")
                st.markdown("• **Strength Power 5**: If the last year is from a school in the formerly Power 5")
                
                st.markdown("**Combine/Athletic Metrics:**")
                st.markdown("• **Combine metrics**: Various standard combine measurement metrics (height, weight, speed, etc.)")

        st.session_state.rb_current_prospects = pd.read_csv("rb_data/current_year_rb.csv")
        st.subheader("Current Year Predictions")
        st.dataframe(st.session_state.rb_current_prospects, use_container_width=True)

        st.subheader("Specific Player Distribution")
        st.session_state.specific_rb = st.selectbox("Select a player to see their distribution of their min, max, and predicted values. These are generated from the stepping around their actual combine numbers. If a player's prediction is near their max it means that they wouldn't be considered any better if they were a better athlete than they tested at the combine. If they are near their min, it means that the projections wouldn't get much worse if they were a worse athlete than they tested at the combine.", st.session_state.rb_current_prospects.ID.unique())

        plot_df = st.session_state.rb_current_prospects[st.session_state.rb_current_prospects.ID == st.session_state.specific_rb]
        row = plot_df.copy()
        # Parameters
        mean = row['predicted_value'].item()
        min_val =  row['min_value'].item()
        max_val =  row['max_value'].item()
        player = st.session_state.specific_rb

        distance_to_min = mean - min_val
        distance_to_max = max_val - mean
        max_distance = max(distance_to_min, distance_to_max)
        std_dev_estimate2 = max_distance / 3  # Assuming max distance is 3 standard deviations

        # Choose appropriate std_dev estimation method
        std_dev = std_dev_estimate2  # Change to std_dev_estimate2 if needed

        # Generate x values for plotting
        x = np.linspace(min_val - std_dev, max_val + std_dev, 1000)

        # Calculate normal PDF
        y = stats.norm.pdf(x, mean, std_dev)

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot the normal distribution
        plt.plot(x, y, 'r-', lw=2, label='Normal Distribution')

        # Shade the area between min and max
        plt.fill_between(x, y, where=(x >= min_val) & (x <= max_val), 
                        color='skyblue', alpha=0.4, label='Min-Max Range')

        # Add vertical lines for min, max, and mean
        plt.axvline(min_val, color='green', linestyle='--', alpha=0.7, label=f'Minimum = {min_val}')
        plt.axvline(max_val, color='purple', linestyle='--', alpha=0.7, label=f'Maximum = {max_val}')
        plt.axvline(mean, color='black', linestyle='-', alpha=0.7, label=f'Mean = {mean}')

        # Add labels and title
        plt.xlabel('Predicted_Output')
        plt.ylabel('Probability Density')
        plt.title(f'{player} : Normal Distribution with Min={min_val:.2f}, Mean={mean:.2f}, Max={max_val:.2f}, and Estimated Std Dev={std_dev:.2f}')
        plt.grid(True, alpha=0.3)


        plt.xlim(0,100)
        # Show the plot
        plt.tight_layout()
        # Show the plot
        st.pyplot(plt)

        
        # Read in the upcoming draft table
        st.session_state.rb_full_prospects = pd.read_csv("rb_data/full_list_prospects_rb.csv")
        st.subheader("Full Predictions")
        st.write('Here is the full table for all players in the dataset along with their NFL target variable to see the biggest misses in both directions. I will touch more on the specific misses below the table.')
        st.dataframe(st.session_state.rb_full_prospects, use_container_width=True)
        st.write('The biggest underpredictions were Rico Dowdle, Alexander Mattison, Isaih Pacheco, Damien Harris, and Corey Clement. Notable names also in the top 15 are JK Dobbins, Kareem Hunt, Breece Hall, Dalvin Cook and Josh Jacobs.')
        st.write("A handful of the biggest misses were lower drafted players who produced on the field though might not have been long term contributors or foundational pieces for a team. Though for some of the bigger misses, they at least had a higher max_value prediction to highlight their ceiling could be higher in the pros.")

        st.write('On the flip side of the coin, biggest overpredictions of note were Jonathon Brooks, Kimani Vidal, Royce Freeman, Deuce Vaughn, Todd Gurley, and Rachaad White.')
        st.write("It is worth noting that some of those players are rookies and might not maintain their current level of play. As they say, you're never wrong, just early. Other than that there are some undersized players and players who just haven't gotten a shot. It is harder to demand carries unless you're really good in the NFL because you typically only have one RB on the field compared to having multiple WR on the field to earn targets.")


        st.subheader("Model Process and Explanations")
        st.write("Below I am going to talk out the different choices I made during the modeling process and why. Then I'll show some of the model evaluation with SHAP plots, correlation plots, and error metrics.")
        st.write("The first choice that I made was to determine what I was going to use as the target variable since whatever you tell the model to predict, it will. The target variable is therefore a very important decision. I chose to create a composite metric that is a weighted average of a handful of features that I use and have found to be very indicative of NFL success. These featres are Attempts, Forced missed tackle rate, Explosive rush rate, touchdown per carry, yards after contact per carry, yards per route run, and Wins above replacement. These values are converted to percentiles and scaled according to the following weights: Attempts = 15, ForcedMissedTackleRate = 5, ExplosiveRate = 15, TDP = 5, YardsAfterContact = 10, YPRR = 5, WAR = 5. I found that to be a good mix of volume and efficiency metrics and guys like Jahmyr Gibbs, Nick Chubb, Alvin Kamara, Adrian Peterson, and Bijan Robinson dominate the top of the list.")
        st.write("I also filtered this to only include a player's first 4 seasons in the NFL to try and capture as many players as possible and avoid aging curves and survivorship bias. I adjusted down the value of a first season for a player to be 50% weighted toward the average first year player. I then created a scaling feature using the logit function to weight the model more towards predicting the top end running backs correctly, while putting less emphasis on the lower value players.")
        st.write("I created the college data using the values as described above and I created features for best and worst features for an individual season to capture if someone had a breakout year that wasn't captured in the raw average. I also imputed the combine data for players that didn't have data listed by using their production scores with a KNN to gather their athleticism data. I found this to boost the model quite a bit for individual projections.")
        st.write("Then I removed players who weren't drafted to ensure that the model wasn't training on guys who played in college but weren't legit prospects. I then created monotonic constraints for the features so the model learned that being faster is better for example and not try to find pockets of 40 yard dash times with higher value. I don't want the model to learn that running a 4.45 is better than a 4.37 just because some guys who ran a 4.37 weren't good in the pros.")
       

        st.subheader("Model Evaluation")
        st.write("I trained an XGBoost model with the following hyperparameters [learning_rate: 0.05, max_depth: 5, num_estimators: 100, col_subsample: 0.8, subsample: 0.8].")
        st.write("The model performed pretty well on the test set with the following metrics [R^2: 0.220, rmse: 0.187, mae: 0.145]. Below is the correlation plot between the predicted and actual target for the entire sample outside of the 2024 prospects.")
        st.image("rb_data/correlation_scatter_plot.png", caption="Scatter Plot of Predicted vs Actual with line of best fit.")

        st.write("I also created a SHAP analysis of the model with the beeswarm and bar plots showing the overall model features and their importance and the waterfall plot showing an example with the first row in the sample to show how the prediction is made.")
        st.image("rb_data/shap_bar_plot.png", caption="Bar Plot of shap values for the features in the model")
        st.image("rb_data/shap_beeswarm_plot.png", caption="Beeswarm plot for shap values and the corresponding feature values")
        st.image("rb_data/better_waterfall_plot.png", caption="Waterfall plot showing the model prediction for the first player in the sample")


        st.session_state.rb_shap_column = st.selectbox("Select the specific feature to see how the SHAP values change across that feature", sorted(set(st.session_state.rb_full_prospects.columns)))
        st.image(f"rb_data/shap_scatter_plot_{st.session_state.rb_shap_column}.png", caption="Shap Scatter plot showing the shap values for a specific feature and the more correlated secondary feature")





st.subheader("About Me")
st.write('If you want to know more about me and my work: Check out my [Portfolio](https://sites.google.com/view/seth-lanza-portfolio/home), Get in touch at Seth.Lanza@gmail.com, Connect on [Linkedin](https://www.linkedin.com/in/sethlanza/), Check my [Twitter](https://x.com/SethDataScience), or Check out my [Github](https://github.com/SethTheDataScientist?tab=repositories)')