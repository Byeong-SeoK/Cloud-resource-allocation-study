import ast
from collections import Counter

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

#그룹 내에 인스턴스가 몇개 들어있고 그 중에서 할당할 인스턴스로 이것을 선택했다는 뉘앙스의 출력 추가 필요

# 데이터 불러오기
tf_data = pd.read_csv('./tensorflow_grouped_resource_usage.csv')
pytorch_data = pd.read_csv('./pytorch_grouped_resource_usage.csv')
graphlearn_data = pd.read_csv('./graphlearn_grouped_resource_usage.csv')
jupyter_data = pd.read_csv('./Jupyter_grouped_resource_usage.csv')

# 데이터를 dictionary로 구성
data = {
    'tensorflow': tf_data,
    'pytorch': pytorch_data,
    'graphlearn': graphlearn_data,
    'Jupyter': jupyter_data
}

# 사용자 입력 값
input_cpu = 900.0
input_mem = 29.2974
input_gpu = None

task_name = None
group_id = None

cpu = None
mem = None
gpu = None


if input_gpu is not None:
    # GPU 값이 입력된 경우
    min_diff = float('inf')
    nearest_group_id = None
    nearest_csv = None

    features = jupyter_data[['plan_cpu', 'plan_mem', 'plan_gpu']]
    kmeans_jupyter = KMeans(n_clusters=4, n_init=10, random_state=42)
    jupyter_data['cluster'] = kmeans_jupyter.fit_predict(features)

    nn_jupyter = NearestNeighbors(n_neighbors=4)
    nn_jupyter.fit(features)

    # feature의 이름을 명시
    input_data = pd.DataFrame([[input_cpu, input_mem, input_gpu]], columns=['plan_cpu', 'plan_mem', 'plan_gpu'])
    input_cluster_jupyter = kmeans_jupyter.predict(input_data)[0]

    distance_jupyter, min_diff_row_index_jupyter = nn_jupyter.kneighbors(input_data)
    predicted_cluster_jupyter = jupyter_data.at[min_diff_row_index_jupyter[0][0], 'cluster']

    if distance_jupyter[0][0] < min_diff:
        min_diff = distance_jupyter[0][0]
        nearest_group_id = min_diff_row_index_jupyter[0][0]
        nearest_csv = 'Jupyter'

    print(f"Task name: {nearest_csv}")
    print(f"The closest group: {nearest_group_id}")

    task_name = nearest_csv
    group_id = nearest_group_id

    # 추가로 출력하는 부분
    # print(f"The closest row information:")
    # print(jupyter_data.iloc[nearest_group_id])
    cpu = data[nearest_csv].iloc[nearest_group_id][2]
    mem = data[nearest_csv].iloc[nearest_group_id][3]
    gpu = data[nearest_csv].iloc[nearest_group_id][4]

else:
    min_diff = float('inf')
    nearest_group_id = None
    nearest_csv = None

    for csv_name, csv_data in data.items():
        if csv_name == 'Jupyter':
            continue  # Jupyter는 위에서 처리했으므로 건너뜁니다.

        features = csv_data[['plan_cpu', 'plan_mem']]
        kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
        csv_data['cluster'] = kmeans.fit_predict(features)

        nn = NearestNeighbors(n_neighbors=4)
        nn.fit(features)

        # feature의 이름을 명시
        input_data = pd.DataFrame([[input_cpu, input_mem]], columns=['plan_cpu', 'plan_mem'])
        input_cluster = kmeans.predict(input_data)[0]

        distance, min_diff_row_index = nn.kneighbors(input_data)
        predicted_cluster = csv_data.at[min_diff_row_index[0][0], 'cluster']

        if distance[0][0] < min_diff:
            min_diff = distance[0][0]
            nearest_group_id = min_diff_row_index[0][0]
            nearest_csv = csv_name

    print(f"Task name: {nearest_csv}")
    print(f"The closest group: {nearest_group_id}")

    task_name = nearest_csv
    group_id = nearest_group_id

    # 추가로 출력하는 부분
    # print(f"The closest group information:")
    #print(data[nearest_csv].iloc[nearest_group_id])
    cpu = data[nearest_csv].iloc[nearest_group_id][2]
    mem = data[nearest_csv].iloc[nearest_group_id][3]




def get_inst_id_from_grouped_data(task_name, cpu_value, mem_value, gpu_value):
    if (task_name == "tensorflow"):
        csv_file_path = "./tensorflow_grouped_data.csv"

    elif (task_name == 'pytorch'):
        csv_file_path = "./pytorch_grouped_data.csv"

    elif (task_name == 'graphlearn'):
        csv_file_path = "./graphlearn_grouped_data.csv"

    elif (task_name == 'Jupyter'):
        csv_file_path = "./Jupyter_grouped_data.csv"


    # 주어진 task_name에 해당하는 CSV 파일 불러오기
    task_data = pd.read_csv(csv_file_path)

    # 주어진 task_name으로부터 얻은 cpu, mem, gpu 값과 동일한 값을 가진 행 찾기
    if(task_name == 'Jupyter'):
        matching_rows = task_data[(task_data['plan_cpu'] == cpu_value)
                                  & (task_data['plan_mem'] == mem_value)
                                  & (task_data['plan_gpu'] == gpu_value)]
    else:
        matching_rows = task_data[(task_data['plan_cpu'] == cpu_value)
                                  & (task_data['plan_mem'] == mem_value)]

    # 찾은 행이 있으면 그 중 임의의 행을 선택하여 inst_id 값 출력
    if not matching_rows.empty:
        selected_row = matching_rows.sample(n=1, random_state=42)
        inst_num = selected_row['inst_id_count'].iloc[0]
        inst_id_value = ast.literal_eval(selected_row['inst_id'].iloc[0])[0]

        instance_ids_list = ast.literal_eval(inst_id_value)

        # 중복 제거 후 유니크한 값만 추출
        unique_instance_ids = list(set(instance_ids_list))


        print(f"Number of instances in the group: {inst_num}")
        print(f"Instance ID to allocate in cluster: {unique_instance_ids}")

    else:
        print("No matching cluster found.")

# 주어진 task_name, group_id, cpu, mem, gpu를 이용하여 함수 호출
get_inst_id_from_grouped_data(task_name, cpu, mem, gpu)



def calculate_accuracy(input_cpu, cpu, input_mem, mem, input_gpu=None, gpu=None):
    # 여기에 값들 간의 차이 계산 및 정확도 측정 로직을 구현하세요
    diff_cpu = abs(input_cpu - cpu)
    diff_mem = abs(input_mem - mem)
    diff_gpu = 0

    if(input_gpu is not None):
        diff_gpu = abs(input_gpu - gpu)

    # 예시로 각 차이의 절대값을 평균하여 정확도 측정
    accuracy = 100 - (diff_cpu + diff_gpu + diff_mem) / 3

    return accuracy


if input_gpu is not None:
    accuracy = calculate_accuracy(input_cpu, cpu, input_mem, mem, input_gpu, gpu)
    print(f"Accuracy: {accuracy}%")

else:
    accuracy = calculate_accuracy(input_cpu, cpu, input_mem, mem, input_gpu, gpu)
    print(f"Accuracy: {accuracy}%")