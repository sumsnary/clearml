from clearml import Task
task = Task.init(project_name="my project", task_name="my task")
from clearml import Task, OutputModel, StorageManager
import functools
import inspect
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


def clearml_task(project_name, task_name=None, tags=None, upload_artifacts=None, log_images=None,
                 output_model_name=None):
    """
    Декоратор для автоматической интеграции с ClearML.

    Args:
        project_name (str): Имя проекта в ClearML.
        task_name (str, optional): Имя задачи. Если не указано, берется имя функции.
        tags (list, optional): Список тегов для задачи.
        upload_artifacts (list, optional): Список путей к файлам для загрузки как артефакты.
        log_images (list, optional): Список путей к изображениям для логирования в ClearML.
        output_model_name (str, optional): Имя выходной модели.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Автоматическое определение имени задачи
            automatic_task_name = task_name or func.__name__

            # Создание задачи ClearML
            task = Task.init(
                project_name=project_name,
                task_name=automatic_task_name,
                tags=tags or []
            )

            # Логирование параметров функции
            task.connect_configuration(
                {k: v for k, v in inspect.signature(func).parameters.items()},
                name="function_signature"
            )

            output_model = None
            if output_model_name:
                output_model = OutputModel(task=task, name=output_model_name)

            try:
                # Выполнение основной функции
                result = func(*args, **kwargs)

                # Логирование результатов успешного выполнения
                task.get_logger().report_scalar(
                    title='Execution',
                    series='Success',
                    value=1
                )

                # Загрузка артефактов (если указано)
                if upload_artifacts:
                    for artifact_path in upload_artifacts:
                        if os.path.exists(artifact_path):
                            task.upload_artifact(name=os.path.basename(artifact_path), artifact_object=artifact_path)
                        else:
                            task.get_logger().report_text(f"Warning: Artifact file not found: {artifact_path}")
                # Логирование изображений (если указано)
                if log_images:
                    for image_path in log_images:
                        if os.path.exists(image_path):
                            task.get_logger().report_image(title=os.path.basename(image_path),
                                                           series="images",
                                                           local_path=image_path)
                        else:
                            task.get_logger().report_text(f"Warning: Image file not found: {image_path}")

                if output_model:
                    output_model.update_weights(model_path=result)

                return result

            except Exception as e:
                # Обработка ошибок
                task.get_logger().report_scalar(
                    title='Execution',
                    series='Error',
                    value=1
                )
                task.get_logger().report_text(str(e))
                raise

            finally:
                pass  # task.close() убрал т.к Task.init создает объект в контексте, который сам закроется, чтобы избежать ошибок

        return wrapper

    return decorator


def load_data(file_path):
    """Загрузка данных из CSV файла."""
    return pd.read_csv(file_path)


@clearml_task(
    project_name="my project",
    task_name="my task",
    tags=["preprocessing", "v2", "extended"],
    upload_artifacts=["processed_data.csv"],
    log_images=["data_histogram.png"],
    output_model_name='my_model_test'
)
def process_data_extended(input_file, threshold=0.5):
    """Обработка данных с сохранением артефакта и генерацией графика."""
    # Ваш код обработки данных
    data = load_data(input_file)
    processed_data = data[data['score'] > threshold]
    processed_data.to_csv("processed_data.csv", index=False)

    # Пример генерации и сохранения графика
    plt.figure(figsize=(8, 6))
    plt.hist(processed_data['score'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of Processed Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig('data_histogram.png')
    plt.close()

    # сохранение модели (в данном случае путь к файлу)
    # output_model_name будет сохранен только путь к файлу (на самом деле нужно сохранять веса модели)
    return "processed_data.csv"


if __name__ == '__main__':
    # Создадим тестовый dataset.csv
    data = {'score': np.random.rand(100)}
    df = pd.DataFrame(data)
    df.to_csv('dataset.csv', index=False)

    # Использование декоратора
    result = process_data_extended('dataset.csv', threshold=0.7)
    print(f"Результат: {result}")

    # Удаление тестовых файлов
    os.remove('dataset.csv')
    os.remove('processed_data.csv')
    os.remove('data_histogram.png')
