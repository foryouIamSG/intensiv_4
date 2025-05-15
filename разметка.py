import pandas as pd
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field 
from typing import List, Any, Union, Optional, Tuple, Callable
load_dotenv()

class FeedbackCategories(BaseModel):

    comment: str = Field(..., description="Исходный комментарий")
    rating: float = Field(..., description="Оценка пользователя")
    speed_satisfaction: int = Field(
        default=0,
        description="Нравится скорость отработки заявок",
        ge=0,
        le=1
    )
    quality_satisfaction: int = Field(
        default=0,
        description="Нравится качество выполнения заявки",
        ge=0,
        le=1
    )
    staff_satisfaction: int = Field(
        default=0,
        description="Нравится качество работы сотрудников",
        ge=0,
        le=1
    )
    request_satisfaction: int = Field(
        default=0,
        description="Понравилось выполнение заявки",
        ge=0,
        le=1
    )
    issue_resolved: int = Field(
        default=0,
        description="Вопрос решен",
        ge=0,
        le=1
    )
    job_not_complete:int = Field(
        default=0,
        description="Работа не выполненена",
        ge=0,
        le=1

    )



#по гайду OpenAI 
client = OpenAI(base_url='http://127.0.0.1:1234/v1', api_key='lm-studio')

def analyze_comment(comment: str, rating: float) -> FeedbackCategories:
    prompt = f"""
    Проанализируй комментарий и его рейтинг, присвой значения от 0 до 1 категориям:
    1. Нравится скорость отработки заявок
    2. Нравится качество выполнения заявки
    3. Нравится качество работы сотрудников
    4. Понравилось выполнение заявки
    5. Вопрос решен
    6. Работа не выполненена

    Comment: {comment}
    Rating: {rating}

    Верни JSON объект в следующем формате:
    {{
        "comment": "{comment}",
        "rating": {rating},
        "speed_satisfaction": 0 или 1,
        "quality_satisfaction": 0 или 1,
        "staff_satisfaction": 0 или 1,
        "request_satisfaction": 0 или 1,
        "issue_resolved": 0 или 1,
        "job_not_complete": 0 или 1
    }}
    """

    response = client.beta.chat.completions.parse(
        model="ruadapt_qwen2.5_3b_ext_u48_instruct_v4_gguf",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes customer feedback."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format=FeedbackCategories,
    )

    result: FeedbackCategories = response.choices[0].message.parsed
    return result

def process_csv(input_file: str, output_file: str, output_format: str = 'json'):
    """
    Process CSV file and save results in specified format (json or csv)
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output file
        output_format (str): 'json' or 'csv'
    """
    # читаем файл
    df = pd.read_csv(input_file)
    
    
    results = []
    for index, row in df.iterrows():
        comment = row['comment']  # Adjust column name as needed
        rating = row['rating']    # Adjust column name as needed
        
        analysis = analyze_comment(comment, rating)
        results.append(analysis.model_dump())
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results in specified format
    if output_format.lower() == 'csv':
        results_df.to_csv(output_file, index=False, encoding='utf-8')
    else:  
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = "input.csv"  # файл начальный
    output_file = "output.csv"  # размеченный файл
    process_csv(input_file, output_file, output_format='csv')  