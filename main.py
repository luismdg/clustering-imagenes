import os
import subprocess
import sys
import argparse

def archivo_tiene_contenido(ruta_archivo):
    """Verifica si un archivo existe y tiene contenido (más de 10 bytes)"""
    if not os.path.exists(ruta_archivo):
        return False
    return os.path.getsize(ruta_archivo) > 10

def ejecutar_comparacion(directorio):
    """Ejecuta el script de comparación para un directorio específico"""
    # Ruta base del proyecto
    base_dir = os.path.dirname(os.path.abspath(__file__))
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
                    return True
                else:
                    print(f"ERROR {directorio}/comparacion.py falló con código: {resultado.returncode}")
                    return False
                    
            except Exception as e:
                print(f"Error ejecutando {directorio}/comparacion.py: {e}")
                return False
            finally:
                # Volver al directorio original
                os.chdir(base_dir)
        else:
            print(f"AVISO: {directorio}/comparacion.py existe pero está vacío")
            return False
    else:
        print(f"AVISO: No se encontró: {directorio}/comparacion.py")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Ejecutar comparaciones de métodos de clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos de uso:
  python main.py --gmm              # Ejecutar solo GMM
  python main.py --espectral        # Ejecutar solo Espectral
  python main.py --kmeans           # Ejecutar solo K-Means
  python main.py --all              # Ejecutar todos los métodos
  python main.py --gmm --kmeans     # Ejecutar GMM y K-Means

Si no se especifica ningún argumento, se mostrará este mensaje de ayuda.
        '''
    )
    
    parser.add_argument(
        '--gmm', 
        action='store_true',
        help='Ejecutar comparación para método GMM'
    )
    
    parser.add_argument(
        '--espectral', 
        action='store_true',
        help='Ejecutar comparación para método Espectral'
    )
    
    parser.add_argument(
        '--kmeans', 
        action='store_true',
        help='Ejecutar comparación para método K-Means'
    )
    
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Ejecutar todos los métodos de comparación'
    )
    
    args = parser.parse_args()
    
    # Si no se proporciona ningún argumento, mostrar ayuda
    if not any([args.gmm, args.espectral, args.kmeans, args.all]):
        parser.print_help()
        return
    
    print("=" * 60)
    print("EJECUTANDO COMPARACIONES DE MÉTODOS DE CLUSTERING")
    print("=" * 60)
    
    # Determinar qué métodos ejecutar
    metodos_a_ejecutar = []
    
    if args.all:
        metodos_a_ejecutar = ['gmm', 'espectral', 'kmeans']
        print("Modo: EJECUTAR TODOS LOS MÉTODOS")
    else:
        if args.gmm:
            metodos_a_ejecutar.append('gmm')
        if args.espectral:
            metodos_a_ejecutar.append('espectral')
        if args.kmeans:
            metodos_a_ejecutar.append('kmeans')
        print(f"Modo: MÉTODOS ESPECÍFICOS - {', '.join(metodos_a_ejecutar)}")
    
    resultados = {}
    
    for metodo in metodos_a_ejecutar:
        resultados[metodo] = ejecutar_comparacion(metodo)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    
    exitos = sum(resultados.values())
    total = len(resultados)
    
    for metodo, exito in resultados.items():
        estado = "✅ EXITOSO" if exito else "❌ FALLIDO"
        print(f"{metodo:10} : {estado}")
    
    print(f"\nTotal: {exitos}/{total} métodos ejecutados exitosamente")
    
    if exitos == total:
        print("🎉 ¡Todas las comparaciones completadas exitosamente!")
    elif exitos > 0:
        print("⚠️  Algunas comparaciones se completaron con errores")
    else:
        print("💥 Todas las comparaciones fallaron")
    
    print("=" * 60)

if __name__ == "__main__":
    main()