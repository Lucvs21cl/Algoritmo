import pandas as pd
import numpy as np
import joblib
import os

def obtener_datos_pasajero():
    print("\n=== Ingrese los datos del pasajero ===")
    
    # Datos básicos
    edad = int(input("Edad: "))
    genero = input("Género (Male/Female): ").lower()
    tipo_cliente = input("Tipo de Cliente (Loyal Customer/disloyal Customer): ").lower()
    tipo_viaje = input("Tipo de Viaje (Business travel/Personal Travel): ").lower()
    clase = input("Clase (Business/Eco): ").lower()
    distancia_vuelo = float(input("Distancia del vuelo: "))
    
    # Servicios (0-5)
    print("\nCalifique los siguientes servicios (0-5):")
    wifi = int(input("Servicio de WiFi a bordo: "))
    tiempo_conveniente = int(input("Conveniencia de horario de salida/llegada: "))
    reserva_online = int(input("Facilidad de reserva online: "))
    ubicacion_puerta = int(input("Ubicación de la puerta: "))
    comida = int(input("Comida y bebida: "))
    embarque_online = int(input("Embarque online: "))
    comodidad_asiento = int(input("Comodidad del asiento: "))
    entretenimiento = int(input("Entretenimiento a bordo: "))
    servicio_bordo = int(input("Servicio a bordo: "))
    espacio_piernas = int(input("Servicio de espacio para piernas: "))
    equipaje = int(input("Manejo de equipaje: "))
    checkin = int(input("Servicio de check-in: "))
    servicio_vuelo = int(input("Servicio durante el vuelo: "))
    limpieza = int(input("Limpieza: "))
    
    # Retrasos
    retraso_salida = float(input("Retraso en la salida (minutos): "))
    
    # Crear diccionario con los datos en el orden correcto
    datos = {
        'Age': edad,
        'Gender': 1 if genero == 'male' else 0,
        'Customer Type': 0 if tipo_cliente == 'loyal customer' else 1,
        'Type of Travel': 0 if tipo_viaje == 'business travel' else 1,
        'Class': 0 if clase == 'business' else 1,
        'Flight Distance': distancia_vuelo,
        'Inflight wifi service': wifi,
        'Departure/Arrival time convenient': tiempo_conveniente,
        'Ease of Online booking': reserva_online,
        'Gate location': ubicacion_puerta,
        'Food and drink': comida,
        'Online boarding': embarque_online,
        'Seat comfort': comodidad_asiento,
        'Inflight entertainment': entretenimiento,
        'On-board service': servicio_bordo,
        'Leg room service': espacio_piernas,
        'Baggage handling': equipaje,
        'Checkin service': checkin,
        'Inflight service': servicio_vuelo,
        'Cleanliness': limpieza,
        'Departure Delay in Minutes': retraso_salida,
        'satisfaction': 1
    }
    
    # Crear DataFrame y asegurar el orden de las columnas
    df = pd.DataFrame([datos])
    
    # Ordenar las columnas según el orden del modelo
    columnas_ordenadas = [
        'Age', 'Gender', 'Customer Type', 'Type of Travel', 'Class',
        'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
        'Ease of Online booking', 'Gate location', 'Food and drink',
        'Online boarding', 'Seat comfort', 'Inflight entertainment',
        'On-board service', 'Leg room service', 'Baggage handling',
        'Checkin service', 'Inflight service', 'Cleanliness',
        'Departure Delay in Minutes', 'satisfaction'
    ]
    
    return df[columnas_ordenadas]

def clasificar_pasajero():
    try:
        # Obtener la ruta del directorio actual
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Cargar el modelo y el scaler
        model = joblib.load(os.path.join(current_dir, 'kmeans_model.joblib'))
        scaler = joblib.load(os.path.join(current_dir, 'scaler.joblib'))
        pca = joblib.load(os.path.join(current_dir, 'pca.joblib'))
        columnas = joblib.load(os.path.join(current_dir, 'columnas.joblib'))
        
        # Obtener datos del pasajero
        datos_pasajero = obtener_datos_pasajero()
        
        # Asegurar que las columnas estén en el orden correcto
        datos_pasajero = datos_pasajero[columnas]
        
        # Preprocesar los datos
        datos_escalados = scaler.transform(datos_pasajero)
        datos_pca = pca.transform(datos_escalados)
        
        # Predecir el cluster
        cluster = model.predict(datos_pca)[0]
        
        # Mostrar resultado
        print("\n=== Resultado de la clasificación ===")
        print(f"El pasajero pertenece al Grupo {cluster}")
        
        # Mostrar descripción detallada del grupo y paquete recomendado
        descripciones = {
            0: {
                "nombre": "Viajeros de negocios frecuentes/leales",
                "paquete": "Executive Sky Priority",
                "descripcion": """Pensado para ejecutivos frecuentes y altamente leales, el paquete "Executive Sky Priority" ofrece una experiencia premium desde el primer momento. Incluye acceso ilimitado a salas VIP, check-in y embarque preferente, WiFi premium sin restricciones, y un kit de productividad que contempla cargadores USB, luz de lectura y material de trabajo. Además, permite acumular el triple de millas por cada vuelo y ofrece atención personalizada mediante un concierge exclusivo. Los pasajeros también acceden a upgrades automáticos a clase superior cuando hay disponibilidad, asegurando una experiencia ágil, cómoda y a la altura de sus expectativas."""
            },
            1: {
                "nombre": "Público joven",
                "paquete": "FlySmart Youth",
                "descripcion": """Dirigido a jóvenes viajeros con bajo presupuesto que vuelan por motivos personales, el paquete "FlySmart Youth" busca mejorar su experiencia sin aumentar significativamente el costo. Incluye entretenimiento digital gratuito a bordo, selección de asiento sin cargo adicional, un snack incluido en vuelos de más de 2 horas y descuentos exclusivos para futuros viajes. Además, se activa un programa de referidos ideal para estudiantes o grupos de amigos, y se les permite embarcar juntos sin complicaciones. Es un paquete atractivo, accesible y diseñado para viajeros exigentes con pocos recursos."""
            },
            2: {
                "nombre": "Viajeros de negocios infrecuentes",
                "paquete": "Business Flex Pro",
                "descripcion": """Este paquete está orientado a profesionales que viajan ocasionalmente por trabajo, muchas veces con poca experiencia en vuelos. "Business Flex Pro" ofrece asistencia personalizada 24/7 para facilitar la reserva y cambios de pasajes, recordatorios inteligentes para el check-in y abordaje, y la asignación automática del mejor asiento disponible. Incluye además dos horas de WiFi gratuito a bordo, horarios preferentes para facilitar la asistencia a reuniones, y la opción de contratar transporte corporativo en destino. El enfoque está en reducir la fricción en la logística del viaje y brindar control y confianza al pasajero."""
            },
            3: {
                "nombre": "Viajeros de vacaciones familiares infrecuentes",
                "paquete": "Family Holiday Breeze",
                "descripcion": """Diseñado para familias que viajan por vacaciones, el paquete "Family Holiday Breeze" busca aliviar el estrés de los padres durante el viaje. Ofrece asientos agrupados garantizados sin costos adicionales, embarque prioritario para familias con niños, una maleta infantil extra sin cargo, y snacks y entretenimiento a bordo pensados para menores. Además, incluye asistencia especial en el aeropuerto para facilitar el tránsito con cochecitos, pañales o equipaje voluminoso. Quienes reserven en grupo también acceden a descuentos especiales. Es una solución cómoda, práctica y enfocada en el bienestar familiar."""
            }
        }
        
        grupo = descripciones[cluster]
        print(f"\nTipo de pasajero: {grupo['nombre']}")
        print(f"\nPaquete recomendado: {grupo['paquete']}")
        print("\nDescripción del paquete:")
        print(grupo['descripcion'])
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Asegúrese de que el archivo knn_classifier.py se haya ejecutado primero para generar los modelos.")

def main():
    while True:
        print("\n=== Menú de Clasificación de Pasajeros ===")
        print("1. Clasificar nuevo pasajero")
        print("2. Salir")
        
        opcion = input("\nSeleccione una opción: ")
        
        if opcion == "1":
            clasificar_pasajero()
        elif opcion == "2":
            print("¡Hasta luego!")
            break
        else:
            print("Opción no válida. Por favor, intente de nuevo.")

if __name__ == "__main__":
    main() 