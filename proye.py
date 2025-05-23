import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# size predeterminado de los plots
plt.rcParams['figure.figsize'] = (12, 6)

def cleaning(df):
    """
    Realiza una limpieza básica del DataFrame:
    - Elimina filas con valores nulos en columnas importantes.
    - Convierte columnas a tipos de datos adecuados.
    """
    df = df.copy()  # copia del df para no modificar el original
    
    # normalizamos nombres de columnas para evitar errores de preprocesamiento
    df.columns = df.columns.str.strip().str.lower()
    # mantenimiento para valores nan
    df.replace("NaN", pd.NA, inplace=True)
    
    if "title" not in df.columns:
        posibles_titles = [col for col in df.columns if "title" in col]
        if posibles_titles:
            df.rename(columns={posibles_titles[0]: "title"}, inplace=True)
    
    # columnas relevantes para mi analisis
    columnas_necesarias = ["title", "episodeduration(in minutes)", "genres", "rating"]
    
    # depuración: imprime las columnas antes de eliminar nulos
    print("Columnas actuales:", df.columns.tolist())
    # eliminacion de nulos
    df.dropna(subset=columnas_necesarias, inplace=True)
    # columna de duración a numérico, forzando errores a NaN para limpieza posterior
    df["episodeduration(in minutes)"] = pd.to_numeric(df["episodeduration(in minutes)"], errors="coerce")
    # columna de rating a numérico, similar al paso anterior
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    # elimina filas que tengan valores nulos en cualquier columna, asegurando integridad de datos
    df.dropna(inplace=True)
    return df

def rating_vs_genre(df):
    """
    Analiza la calificación promedio según el género principal.
    """
    # género principal de la columna 'genres' (primer género listado)
    df["generoprincipal"] = df["genres"].apply(lambda x: x.split(",")[0] if isinstance(x, str) else x)
    #  género principal y la calificación promedio (de mauyor a menor)
    promedio_genero = df.groupby("generoprincipal")["rating"].mean().sort_values(ascending=False)
    # plt horizontal para la calificación promedio por género
    sns.barplot(x=promedio_genero.values, y=promedio_genero.index, palette="viridis")
    plt.title("Calificación promedio por género principal")
    plt.xlabel("Calificación promedio")
    plt.ylabel("Género principal")
    plt.show()

def rating_vs_duration(df):
    """
    Grafica la relación entre duración del episodio y calificación.
    """
    # scatterplot para observar la relación entre duración y rating, coloreando por género
    sns.scatterplot(data=df, x="episodeduration(in minutes)", y="rating", hue="genres", legend=False)
    plt.title("Duración vs Calificación")
    plt.xlabel("Duración (min)")
    plt.ylabel("Rating")
    plt.show()

def rating_vs_show(df):
    """
    Muestra los shows mejor calificados.
    """
    # 15 shows con mayor rating para análisis
    df = df.sort_values("rating", ascending=False).head(15)
    # plt horizontal mostrando los ratings de los top 15 shows
    sns.barplot(x="rating", y="title", data=df, palette="coolwarm")
    plt.title("Top 15 shows con mejor calificación")
    plt.xlabel("Calificación")
    plt.ylabel("Show")
    plt.show()

def rating_vs_show_2(df):
    """
    Muestra los shows con peor calificación.
    """
    # 15 shows con menor rating para análisis
    df = df.sort_values("rating", ascending=True).head(15)
    # plt horizontal mostrando los ratings de los top 15 shows
    sns.barplot(x="rating", y="title", data=df, palette="coolwarm")
    plt.title("Top 15 shows con peor calificación")
    plt.xlabel("Calificación")
    plt.ylabel("Show")
    plt.show()

def static_charts(df):
    """
    Genera gráficos estadisticos:
    - Distribución de calificaciones y duración.
    - Frecuencia de géneros principales.
    """
    # histograma con KDE para visualizar cómo se distribuyen las calificaciones
    sns.histplot(df["rating"], bins=10, kde=True)
    plt.title("Distribución de calificaciones")
    plt.xlabel("Calificaciones")
    plt.ylabel("Frecuencia")
    plt.show()

    # histograma para observar la distribución de la duración de episodios
    sns.histplot(df["episodeduration(in minutes)"], bins=10, color='orange')
    plt.title("Distribución de duración de episodios")
    plt.xlabel("Duración (minutos)")
    plt.ylabel(" Frecuencia ")
    plt.show()

    # género principal para conteo
    df["generoprincipal"] = df["genres"].apply(lambda x: x.split(",")[0])
    # conteo y gráfico de barras horizontal con frecuencia de cada género principal
    sns.countplot(y="generoprincipal", data=df, order=df["generoprincipal"].value_counts().index)
    plt.title("Frecuencia de géneros principales")
    plt.xlabel("Cantidad")
    plt.ylabel("Género principal")
    plt.show()

def genre_vs_year(df):
    """
    Muestra:
    - Qué géneros se han publicado más en cada año.
    - En qué año hubo mejor promedio de calificaciones.
    Usa un gráfico de mapa de calor y un gráfico de línea para mayor variedad.
    """

    #  año de inicio desde la columna 'years' usando expresión regular
    df['year_inicio'] = df['years'].astype(str).str.extract(r'(\d{4})')[0]
    #  año extraído a numérico, forzando errores a NaN para filtrar después
    df['year_inicio'] = pd.to_numeric(df['year_inicio'], errors='coerce')

    # se toman los años 2001 y 2019 para análisis consistente, ya que por alguna razon, el año 2020 no aporta mucho
    df = df[(df['year_inicio'] >= 2001) & (df['year_inicio'] <= 2019)]

    # toma el género principal para agrupar
    df['generoprincipal'] = df['genres'].apply(lambda x: x.split(",")[0] if isinstance(x, str) else x)

    # counter de shows por año y género principal, rellenando valores faltantes con 0
    conteo = df.groupby(['year_inicio', 'generoprincipal']).size().unstack(fill_value=0)

    # heatmap para visualizar la cantidad de shows por género y año
    sns.heatmap(conteo.T, cmap="YlGnBu", linewidths=0.5)
    plt.title("Cantidad de shows por género y año")
    plt.xlabel("Año")
    plt.ylabel("Género principal")
    plt.tight_layout()
    plt.show()

    # calificación promedio por año para detectar tendencias temporales
    rating_por_ano = df.groupby('year_inicio')['rating'].mean()

    # plt de líneas para mostrar evolución del rating promedio a lo largo de los años
    sns.lineplot(x=rating_por_ano.index, y=rating_por_ano.values, marker="o")
    plt.title("Promedio de calificación por año")
    plt.xlabel("Año")
    plt.ylabel("Calificación promedio")
    plt.tight_layout()
    plt.show()

def run_analysis(df):
    """
    Ejecuta todas las funciones de análisis en secuencia.
    """
    funciones = [
        rating_vs_genre,
        rating_vs_duration,
        rating_vs_show,
        rating_vs_show_2,
        genre_vs_year,
        static_charts
    ]
    for funcion in funciones:
        funcion(df)


# Ejecutable para las funciones definidas:
df = pd.read_csv("imdb_tvshows.csv.xls")  
df = cleaning(df)  # Limpieza 
run_analysis(df)