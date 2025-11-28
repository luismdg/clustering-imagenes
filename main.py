import os
import subprocess
import sys

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
            print(f"\n🔍 Ejecutando comparación en: {directorio}/")
            print("-" * 40)
            
            try:
                # Cambiar al directorio del script para que las rutas relativas funcionen
                directorio_script = os.path.join(base_dir, directorio)
                os.chdir(directorio_script)
                
                # Ejecutar el script
                resultado = subprocess.run([sys.executable, 'comparacion.py'], 
                                         capture_output=True, text=True)
                
                # Mostrar salida
                if resultado.stdout:
                    print("✅ Salida:")
                    print(resultado.stdout)
                
                if resultado.stderr:
                    print("❌ Errores:")
                    print(resultado.stderr)
                
                if resultado.returncode == 0:
                    print(f"✅ {directorio}/comparacion.py ejecutado exitosamente")
                else:
                    print(f"❌ {directorio}/comparacion.py falló con código: {resultado.returncode}")
                    
            except Exception as e:
                print(f"❌ Error ejecutando {directorio}/comparacion.py: {e}")
            finally:
                # Volver al directorio original
                os.chdir(base_dir)
        else:
            print(f"⚠️  No se encontró: {directorio}/comparacion.py")
    
    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    ejecutar_comparaciones()