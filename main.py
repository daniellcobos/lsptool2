
from flask import Flask, render_template, jsonify, request, session, url_for
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import json
from os import path
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


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
            crec.append(df.iloc[index].iloc[0])
            for i in range(len(row)):
                if i > 0:
                    # print(row.iloc[i])
                    a = df.iloc[index].iloc[i]
                    b = df.iloc[index - 1].iloc[i]
                    c = ((a / b - 1))
                    crec.append(c*100)
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

def recalculateTotals(df,df_cre,totalizar,changedcolumn):
    firstrow = df.iloc[0]
    dfcre2 = df_cre.copy(deep=True)
    dfcre2 = dfcre2.to_numpy()
    columns = df.columns.tolist()
    df2list = [firstrow]
    if totalizar == 'true':
        for index, row in enumerate(dfcre2):
            try:
                row[1:] = (row[1:]/100) + 1
                nextrow = df2list[index] * row
                nextrow.iloc[0] = row[0]
                bcolumns = columns[1:-1]
                #Toallas es un agregado, se remueve de los calculos
                bcolumns.remove('TOALLAS')
                #saca diferencias


                dif =  sum(nextrow[bcolumns].tolist()) - nextrow.iloc[-1]
                #columnas base
                bcolumns.remove(columns[changedcolumn])
                #peso de las columnas que quedan
                totalq = sum(nextrow[bcolumns].tolist())
                pesosq = nextrow[bcolumns]/totalq
                pesosq = pesosq * dif
                #distribuye diferencias
                nextrow[bcolumns] = nextrow[bcolumns] - pesosq
                nextrow['TOALLAS'] = nextrow['Nocturnas'] + nextrow['Normales'] + nextrow['Ultradelgadas']
                df2list.append(nextrow)
            except Exception as e:
                #print(df_cre.iloc[index])
                print(e)
        df2 = pd.DataFrame(data=df2list, columns=df.columns, index=range(0, len(df2list)))
        return df2
    else:
        for index, row in enumerate(dfcre2):
            try:
                row[1:] = (row[1:]/100) + 1
                nextrow = df2list[index] * row
                nextrow.iloc[0] = row[0]
                df2list.append(nextrow)
            except Exception as e:
                #print(df_cre.iloc[index])
                print(e)

        df2 = pd.DataFrame(data=df2list, columns=df.columns, index= range(0,len(df2list)))
        df2['TOALLAS'] = df2['Nocturnas'] + df2['Normales'] + df2['Ultradelgadas']
        df2['Total General'] = df2["TOALLAS"] + df2["PROTECTORES"] + df2["TAMPONES"]
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
    totalizar = parameters["totalizar"]
    type = parameters["typechange"]
    print(indexchange[0], indexchange[1])
    if type == "foward":
        rows = df.shape[0]
        for r in range(indexchange[0],rows-1):
            try:
                df_cre.iloc[r,indexchange[1]] = float(valuechange)
            except Exception as e:
                print(e)

    df_cre.iloc[indexchange[0],indexchange[1]] = float(valuechange)

    df2 = recalculateTotals(df,df_cre,totalizar,indexchange[1])
    df_cre3 = getCrecimientos(df2)
    session['df'] = df2.to_dict(index=True)

    df2 = df2.to_json(orient='records')

    df_cre3 = df_cre3.to_json(orient='records')

    return [df2,df_cre3]

@app.route('/reset',methods=["POST"])
def reiniciar():
    # Read Excel file
    apppath = app.root_path
    dfpath = path.join(apppath, 'dt.xlsx')
    df = pd.read_excel(dfpath, sheet_name='Totales')
    session['df'] = df.to_dict(index=True)
    return {"resetead":"resetead"}

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def proyeccioninicial(dt):
    # Convierte los periodos a fecha, proyeccion a 5 años
    pproyeccion = 72
    df0 = dt.copy(deep = True)
    df0['Periodo'] = pd.to_datetime(df0['Periodo'], format='%Y-%m-%d')
    tfechaarr = []
    tcrecarr = []
    tcolumnamearr = []
    figdict = {}
    # Determina las columnas y hace la proyeccion
    for (columnName, columnData) in df0.items():

        #Proyeccion inicial, todos tienen los mismos parametros
        ptrend = 'add'
        pseasonal = 'add'
        pdamped = True
        puse = True
        periodos = 12




        if (columnName != 'Periodo'):
            print(columnName)
            df10 = df0[df0[columnName] > 0]
            df10 = df10[['Periodo', columnName]]
            df10 = df10.set_index('Periodo')
            # fig = components.plot()
            # fig.set_size_inches(12, 6)
            # Proyeccion
            fit1 = ExponentialSmoothing(df10, seasonal_periods=periodos, trend=ptrend, seasonal=pseasonal,
                                        damped_trend=pdamped, use_boxcox=puse, initialization_method="estimated").fit()
            # print(fit1.summary())
            # Calcula el intervalo de confianza al 95% y lo salva
            xhat = fit1.forecast(steps=pproyeccion)

            z = 1.96
            sse = fit1.sse

            predint_xminus = xhat - z * np.sqrt(sse / len(df10))
            predint_xplus = xhat + z * np.sqrt(sse / len(df10))

            df = [predint_xminus, predint_xplus]
            df = pd.DataFrame(df)
            df = df.transpose()
            col0 = columnName + "_minimo"
            col1 = columnName + "_maximo"

            # Changing columns name with index number
            mapping = {df.columns[0]: col0, df.columns[1]: col1}
            df = df.rename(columns=mapping)

            # Changing columns name with index number
            mapping = {df.columns[0]: col0, df.columns[1]: col1}
            df = df.rename(columns=mapping)

            # Salva los intervalos


            # Intervalo de confianza por simulacion
            simulations = fit1.simulate(pproyeccion, repetitions=1, error="add", random_errors=None)
            # ax = df10.plot(figsize=(12, 5),marker="o",color="black",title="Proyeccion",)
            # ax.set_ylabel("Venta")
            # ax.set_xlabel("Periodo")
            # fit1.fittedvalues.plot(ax=ax, style="--", color="green")
            graphforecast = fit1.forecast(steps=pproyeccion)
            # graphforecast.rename("Proyeccion").plot(ax=ax, style="--", marker="o", color="green")

            # Grafica la serie de Ajuste y la añade a fit1
            df6 = fit1.forecast(pproyeccion)
            df7 = fit1.predict(0)

            df8 = [df7, df6]
            df10a = df10[columnName]
            df15 = pd.concat([df10a, df6])
            d = {'fecha': df15.index, 'crecimiento': df15[:].tolist()}
            tfechaarr.append(df15.index.tolist())
            tcrecarr.append(df15[:])
            tcolumnamearr.append([str(columnName) for x in df15])
            df16 = pd.DataFrame(data=d)
            fig2 = px.line(df16, x='fecha', y='crecimiento', title=columnName)

            fig2json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
            figdict[columnName] = fig2json
            df1 = (df0[[columnName]])

            df = pd.DataFrame(df8)
            df = df.transpose()

            col0 = columnName + "_ajuste"
            col1 = columnName + "_proyeccion"

            # Changing columns name with index number
            mapping = {df.columns[0]: col0, df.columns[1]: col1}
            df = df.rename(columns=mapping)
            # Salva la proyeccion normal
            mfila = 'py_' + columnName + '.xlsx'
            df.to_excel(mfila, index_label="Periodo")

            # Calcula el Error por medio del MAPE
            df99 = [df7]
            yy = pd.DataFrame(df99)
            yy = yy.transpose()
            xx = pd.DataFrame(df10[[columnName]])
            print("MAPE " + columnName + " = " + str(mape(xx, yy)))

    import itertools

    tfechaarr = list(itertools.chain.from_iterable(tfechaarr))
    tcrecarr = list(itertools.chain.from_iterable(tcrecarr))
    tcolumnamearr = list(itertools.chain.from_iterable(tcolumnamearr))
    totaldata = {'fecha': tfechaarr, 'crecimiento': tcrecarr, 'categoria': tcolumnamearr}
    dftd = pd.DataFrame(data=totaldata)
    fig3 = px.line(dftd, x='fecha', y='crecimiento', title='Crecimientos por Categoria', color='categoria')
    figdict["final"] = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    return figdict

@app.route('/cargarDT')
def cargarDt():
    return render_template("loaddt.html", action=url_for('Proyectar'))


@app.route('/proyectar',methods=["POST"])
def Proyectar():
    uploaded_file = request.files['archivo1']
    if uploaded_file.filename != '':
        mpath = path.join(app.root_path, 'static', 'uploads', uploaded_file.filename)
        uploaded_file.save(mpath)
        dt = pd.read_excel(mpath, sheet_name='dt')
        try:
            figs = proyeccioninicial(dt)

        except Exception as e:
            print(e)
            return "Archivo no valido"
    return render_template("Proyinicial.html", figs = figs)

@app.route('/modificarproyeccion',methods=["POST"])
def modifyproyeccion():
    print(request.json)
    return {"res":"e"}


if __name__ == '__main__':
    app.run(debug=True)
