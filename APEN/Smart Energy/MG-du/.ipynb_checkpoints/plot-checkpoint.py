from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)

def plot(dataframe):
    iplot([{'x':dataframe.index, 'y':dataframe[column],'name':column} for column in dataframe.columns])