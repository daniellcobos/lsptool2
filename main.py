
from flask import Flask, render_template, jsonify,request,session
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
import json
from os import path
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
from datetime import date, datetime
from dateutil.relativedelta import relativedelta



app = Flask(__name__)
app.secret_key = "askljdklajsdsakljdjsakdasdjasd"
@app.route('/')
@app.route('/tendencia')
def tendencia():
    # Read Excel file
    apppath = app.root_path
    dfpath = path.join(apppath,'dt.xlsx')
    df = pd.read_excel(dfpath, sheet_name='dt')
    # Convierte los periodos a fecha
    df['Periodo'] = pd.to_datetime(df['Periodo'], format='%Y-%m-%d')
    
    # Grafica Mensual
    df_grafica = df.copy()
    # Create traces
    fig = go.Figure()
    for (columnName, columnData) in df_grafica.items():
        if columnName != "Periodo" :
            fig.add_trace(go.Scatter(x=df_grafica['Periodo'] , y=df_grafica[columnName] ,
                                mode= 'lines',
                                name= columnName))

    fig = go.Figure(fig) 
    fig.update_layout(title = 'Evolución de la Categoria por Periodos',
     xaxis = dict(
         showgrid = True,                
        ),
     yaxis = dict(
                fixedrange = True,
                showgrid = True,                
        )
    )
    graphJSON1 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Grafica Anual
    df1 = df.groupby(pd.Grouper(key='Periodo', axis=0,freq='YE')).sum() 
    df1.reset_index(inplace = True)
    df1['Anual'] =  df1['Periodo'].dt.year
    df_grafica = df1.copy()
    df_grafica.reset_index(inplace = True)
    # Create traces
    fig = go.Figure()
    for (columnName, columnData) in df_grafica.items():    
        if columnName != "Periodo" and columnName != "Anual"  and columnName != "index" :
            fig.add_trace(go.Scatter(x=df_grafica['Anual'] , y=df_grafica[columnName] ,
                                mode= 'lines',
                                name= columnName))

    fig = go.Figure(fig) 
    fig.update_layout(title = 'Evolución de la Categoria Anual',
     xaxis = dict(
         showgrid = True,                
        ),
     yaxis = dict(
                fixedrange = True,
                showgrid = True,                
        )
    )
    fig.update_xaxes(type='category')
    
    graphJSO2 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


    return render_template('tendencia.html', plot1=graphJSON1, plot2=graphJSO2)

@app.route('/proyeccion')
def proyeccion():
    # Read Excel file
    apppath = app.root_path
    dfpath = path.join(apppath, 'dt.xlsx')
    df = pd.read_excel(dfpath, sheet_name='Crecimientos')
    # Convierte los periodos a fecha
    df['Periodo'] = pd.to_datetime(df['Periodo'], format='%Y-%m-%d')
    
    # Grafica Mensual
    df_grafica = df.copy()
    # Create traces
    fig = go.Figure()
    for (columnName, columnData) in df_grafica.items():
        if columnName != "Periodo" :
            fig.add_trace(go.Scatter(x=df_grafica['Periodo'] , y=df_grafica[columnName] ,
                                mode= 'lines',
                                name= columnName))
            
    fig = go.Figure(fig) 
    fig.update_layout(title = 'Evolución de la Categoria por Periodos',
     xaxis = dict(
         showgrid = True,                
        ),
     yaxis = dict(
                fixedrange = True,
                showgrid = True,                
        )
    )
    graphJSON1 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Grafica Anual
    df1 = df.groupby(pd.Grouper(key='Periodo', axis=0,freq='YE')).sum() 
    df1.reset_index(inplace = True)
    df1['Anual'] =  df1['Periodo'].dt.year
    df_grafica = df1.copy()
    df_grafica.reset_index(inplace = True)
    # Create traces
    fig = go.Figure()
    for (columnName, columnData) in df_grafica.items():    
        if columnName != "Periodo" and columnName != "Anual"  and columnName != "index" :
            fig.add_trace(go.Scatter(x=df_grafica['Anual'] , y=df_grafica[columnName] ,
                                mode= 'lines',
                                name= columnName))

    fig = go.Figure(fig) 
    fig.update_layout(title = 'Evolución de la Categoria Anual',
     xaxis = dict(
         showgrid = True,                
        ),
     yaxis = dict(
                fixedrange = True,
                showgrid = True,                
        )
    )
    fig.update_xaxes(type='category')
    
    graphJSO2 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


    return render_template('proyeccion.html', plot1=graphJSON1, plot2=graphJSO2)

def getCrecimientos(df):
    creci = []
    for index, row in df.iterrows():
        crec = []
        if index > 0:
            crec.append(df.iloc[index][0])
            for i in range(len(row)):
                if i > 0:
                    # print(row.iloc[i])
                    a = df.iloc[index][i]
                    b = df.iloc[index - 1][i]
                    c = round(((a / b - 1)) * 100, 1)
                    c = ((a / b - 1))
                    crec.append(round(c*100,4))
            creci.append(crec)
    df_cre = pd.DataFrame(data=creci, columns=df.columns)
    return df_cre

@app.route('/sensibilizar')
def sensibilizar():
    # Read Excel file
    apppath = app.root_path
    dfpath = path.join(apppath, 'dt.xlsx')
    df = pd.read_excel(dfpath, sheet_name='Totales')
    session['df'] = df.to_dict(index=True)


    df_cre = getCrecimientos(df)

    df2 = df.to_json(orient='records')
    df_cre2 = df_cre.to_json(orient ='records')

    return render_template('sensibilizar.html', df2 = df2, df_cre2= df_cre2)

def recalculateTotals(df,df_cre):
    firstrow = df.iloc[0]
    dfcre2 = df_cre.copy(deep=True)
    dfcre2 = dfcre2.to_numpy()

    df2list = [firstrow]
    for index, row in enumerate(dfcre2):
        try:
            row[1:] = (row[1:]/100) + 1
            nextrow = df2list[index] * row
            nextrow[0] = row[0]
            #nextrow = df2list[index] * row
            df2list.append(nextrow)
        except:
            print(df_cre.iloc[index])
    df2 = pd.DataFrame(data=df2list, columns=df.columns, index= range(0,len(df2list)))
    return df2

@app.route('/changeDt',methods=["POST"])
def changeDt():
    #get columns from OG dataframe
    apppath = app.root_path
    dfpath = path.join(apppath, 'dt.xlsx')
    dfog = pd.read_excel(dfpath, sheet_name='Totales')
    columns = dfog.columns.tolist()
    #gets working df
    df = pd.DataFrame.from_dict(session['df'],dtype=float)
    df.index = df.index.astype(int)
    df = df[columns]
    df_cre = getCrecimientos(df)
    parameters = request.json
    indexchange = json.loads(parameters["indexchange"])
    valuechange = parameters["valuechange"]
    df_cre.iloc[indexchange[0],indexchange[1]] = float(valuechange)
    df2 = recalculateTotals(df,df_cre)
    session['df'] = df2.to_dict(index=True)
    print(df2)
    df2 = df2.to_json(orient='records')

    df_cre3 = df_cre.to_json(orient='records')

    return [df2,df_cre3]


if __name__ == '__main__':
    app.run(debug=True)
