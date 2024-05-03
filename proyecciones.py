import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import json
import plotly.express as px
import numpy as np
import plotly
import itertools

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



    tfechaarr = list(itertools.chain.from_iterable(tfechaarr))
    tcrecarr = list(itertools.chain.from_iterable(tcrecarr))
    tcolumnamearr = list(itertools.chain.from_iterable(tcolumnamearr))
    totaldata = {'fecha': tfechaarr, 'crecimiento': tcrecarr, 'categoria': tcolumnamearr}
    dftd = pd.DataFrame(data=totaldata)
    fig3 = px.line(dftd, x='fecha', y='crecimiento', title='Crecimientos por Categoria', color='categoria')
    figdict["final"] = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    return figdict

def proyeccionmodify(dt,categoriar,ptrendr,pseasonalr,puser,pdampedr):
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


        if columnName == categoriar:
            ptrend = ptrendr
            pseasonal = pseasonalr
            pdamped = pdampedr
            puse = puser
            periodos = 12





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
    return figdict