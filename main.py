import os
import subprocess
import sys

def archivo_tiene_contenido(ruta_archivo):
    """Verifica si un archivo existe y tiene contenido (más de 10 bytes)"""
    if not os.path.exists(ruta_archivo):
        return False
    return os.path.getsize(ruta_archivo) > 10

def ejecutar_comparaciones():
    # Directorios donde buscar los archivos comparacion.py
    directorios = ['espectral', 'gmm', 'kmeans']
    
    # Ruta base del proyecto
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("EJECUTANDO COMPARACIONES DE TODOS LOS MÉTODOS")
    print("=" * 60)
    
    for directorio in directorios:
        ruta_comparacion = os.path.join(base_dir, directorio, 'comparacion.py')
        
        if os.path.exists(ruta_comparacion):
            if archivo_tiene_contenido(ruta_comparacion):
                print(f"\nEjecutando comparación en: {directorio}/")
                print("-" * 40)
                
                try:
                    # Cambiar al directorio del script para que las rutas relativas funcionen
                    directorio_script = os.path.join(base_dir, directorio)
                    os.chdir(directorio_script)
                    
                    # Configurar variables de entorno para soportar Unicode en Windows
                    env = os.environ.copy()
                    env['PYTHONIOENCODING'] = 'utf-8'
                    
                    # Ejecutar el script con codificación UTF-8
                    resultado = subprocess.run(
                        [sys.executable, 'comparacion.py'], 
                        capture_output=True, 
                        text=True,
                        encoding='utf-8',
                        errors='replace',  # Reemplazar caracteres problemáticos
                        env=env
                    )
                    
                    # Mostrar salida
                    if resultado.stdout:
                        print("Salida:")
                        print(resultado.stdout)
                    
                    if resultado.stderr:
                        print("Errores:")
                        print(resultado.stderr)
                    
                    if resultado.returncode == 0:
                        print(f"OK {directorio}/comparacion.py ejecutado exitosamente")
                    else:
                        print(f"ERROR {directorio}/comparacion.py falló con código: {resultado.returncode}")
                        
                except Exception as e:
                    print(f"Error ejecutando {directorio}/comparacion.py: {e}")
                finally:
                    # Volver al directorio original
                    os.chdir(base_dir)
            else:
                print(f"AVISO: {directorio}/comparacion.py existe pero está vacío, saltando...")
        else:
            print(f"AVISO: No se encontró: {directorio}/comparacion.py")
    
    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    ejecutar_comparaciones()