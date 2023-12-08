import ast
import random
from collections import Counter

import seaborn as sns

from cluster_utils import *

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt



pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 100)
matplotlib.rcParams.update({"font.size": 16,'lines.linewidth': 2.5})
matplotlib.rcdefaults()

DATA_DIR = '../data/'
dfj = get_df(DATA_DIR + 'pai_job_table.csv')
dft = get_df(DATA_DIR + 'pai_task_table.csv')
dfi = get_df(DATA_DIR + 'pai_instance_table.csv')
dfs = get_df(DATA_DIR + 'pai_sensor_table.csv')
dfg = get_df(DATA_DIR + 'pai_group_tag_table.csv')
dfp = get_df(DATA_DIR + 'pai_machine_spec.csv')

# dfa = get_dfa(dft, dfj, dfi, dfg)
# dfw = get_dfw(dfi, dft, dfg)

"""==== 필요없는 column 삭제 ===="""
dfc = dft.drop(columns=['start_time', 'end_time', 'gpu_type'])
dfc_new = dft.drop(columns=['status', 'start_time', 'end_time', 'gpu_type'])

dfc_new = dfc_new.set_index("job_name")


"""==== job_name별로 task 묶기 ===="""
task_grouped_df = dfc_new.groupby('job_name')['task_name'].agg(list) #하나의 job에 여러 개의 task 존재
task_grouped_df = task_grouped_df.to_frame()
#task_grouped_df: job name별 task 분류된 csv

# print(1)
# print(task_grouped_df)

"""==== job별로 자원 사용량 묶기 ===="""
dfc_new = dfc_new.drop(columns=['task_name'])
dfc_new = dfc_new.groupby(["job_name"]).mean(['plan_cpu', 'plan_mem', 'plan_gpu'])
#dfc_new: job name별 inst num, plan cpu, plan mem, plan gpu 사용량 평균 csv

# print(2)
# print(dfc_new)



"""==== 같은 그룹의 instance 찾기 ===="""
inst_id_df = dfg.drop(columns=["user", "gpu_type_spec", "workload"])
inst_id_df = inst_id_df.set_index(["inst_id"])
#inst_id_df: inst id별 group을 분류한 csv

# print(inst_id_df)


"""==== inst_id별 task, inst_name 구분 ===="""
inst_by_job_df = dfi.drop(columns=["worker_name", "status", "start_time", "end_time", "machine"])
inst_by_job_df = inst_by_job_df.set_index(["inst_id"])

merge_by_inst_id_df = inst_by_job_df.merge(inst_id_df, how="left", on="inst_id")
merge_by_inst_id_df = merge_by_inst_id_df.reset_index()
#merge_by_inst_id_df: inst id, job name, task name, inst name, group 정보를 가진 csv

merge_by_inst_id_df.to_csv("merge_by_inst_id_df.csv")

#print(3)
#print(merge_by_inst_id_df)


temp = merge_by_inst_id_df.set_index(['job_name'])



"""==== job별로 task묶음과 자원 사용량 묶음 합치기 ===="""
merge_by_task_df = task_grouped_df.merge(dfc_new, how="left", on="job_name")
#merge_by_task_df: job name별 task name, inst num, plan cpu, plan mem, plan gpu 값을 분류한 csv
#다만 여기서 task name이 배열 형태로 되어있어 활용하는 용도로 사용하기는 어렵다.
#또 graph learn이 아닌 [worker, ps] or [ps, worker] 형태로 되어있다.


# print(5)
# print(merge_by_task_df)


"""==== graph learn에 해당하는 것 찾기 ===="""
merge_by_task_df['task_name'] = merge_by_task_df['task_name'].apply(lambda x: ['graph_learn']
    if x == ['ps', 'worker'] or x == ['worker', 'ps'] else x)



"""==== 쓸모없는 데이터 삭제 ===="""
merge_by_task_df = merge_by_task_df[merge_by_task_df['task_name'].apply(len) <= 2]
merge_by_task_df['task_name'] = merge_by_task_df['task_name'].apply(lambda x: x[0] if isinstance(x, list) else x)
#merge_by_task_df: job name별 task name, inst num, plan cpu, plan mem, plan gpu 값을 분류한 csv
#다만 여기서는 task name이 배열 형태가 아니므로 활용할 때는 이 csv를 사용할 것
#또 여기서 드디어 [worker, ps] or [ps, worker]를 graph learn으로 바꿨다.


# print(6)
# print(merge_by_task_df)


# """==== task별 usage 사용량 계산 ===="""
# resource_usage_by_task_df = merge_by_task_df.groupby('task_name').mean(["plan_cpu", "plan_mem", "plan_gpu"])
# resource_usage_by_task_df = resource_usage_by_task_df.drop(columns=["inst_num"])
# resource_usage_by_task_df = resource_usage_by_task_df.dropna()
# resource_usage_by_task_df = resource_usage_by_task_df.reset_index()
# print(resource_usage_by_task_df)






"""==== task별 instance사용량 평균 계산 ===="""
instNum_by_task_df = merge_by_task_df.drop(columns=["plan_cpu", "plan_mem", "plan_gpu"])
instNum_by_task_df = instNum_by_task_df.groupby('task_name').mean(["inst_num"])
#instNum_by_task_df: task name별 inst num(inst 개수) 값을 분류한 csv


# print(7)
#print(instNum_by_task_df)




temp = temp.drop(columns="task_name")
temp = temp.merge(merge_by_task_df, how="left", on="job_name")


result_df = temp.groupby(['job_name', 'group']).agg({
    'inst_id': list,
    'inst_name': list,
    'task_name': list,
    'inst_num': 'mean',
    'plan_cpu': 'mean',
    'plan_mem': 'mean',
    'plan_gpu': 'mean'
}).reset_index()

result_df['task_name'] = result_df['task_name'].apply(lambda x: ['graph_learn']
    if len(x) >= 2 else x)

result_df['task_name'] = result_df['task_name'].apply(lambda x: x[0] if isinstance(x, list) else x)



temp = result_df.drop(columns=["inst_id", "inst_name"])



"""==== PyTorchWorker에 대해서만 데이터 튜닝하려는 로직 ===="""
data = pd.read_csv("./result_df.csv")
filtered_data = data[data['task_name'].apply(lambda x: 'graph_learn' in x)]

grouped_data_for_3D = filtered_data.groupby(['inst_num', 'plan_cpu', 'plan_mem', 'plan_gpu']).agg({
    'inst_id': list,
    'group': list
})
grouped_data_for_3D = grouped_data_for_3D.reset_index()


filtered_data = filtered_data.drop(columns=["plan_gpu"])

grouped_data = filtered_data.groupby(['inst_num', 'plan_cpu', 'plan_mem']).agg({
    'inst_id': list,
    'group': list
})
grouped_data = grouped_data.reset_index()

# print(grouped_data)


#'inst_id'의 개수를 세어 'inst_id_count' 열 생성
grouped_data['inst_id_count'] = grouped_data['inst_id'].apply(lambda x: sum(len(ast.literal_eval(item)) for item in x))
grouped_data.to_csv('graphlearn_grouped_data.csv')


grouped_resource_usage = grouped_data.drop(columns=["inst_id", "group", "inst_id_count"])
grouped_resource_usage.to_csv("graphlearn_grouped_resource_usage.csv")



# 'inst_id_count' 열을 기준으로 내림차순 정렬
sorted_grouped_data = grouped_data.sort_values(by='inst_id_count', ascending=False)

# # 상위 20개 행 선택
filtered = sorted_grouped_data.head(20)


test_set = result_df.drop(columns=["job_name", "inst_name", "group", "inst_id", "task_name", "inst_num"])
test_set = test_set.dropna()
test_set.to_csv("test_set.csv")





"""
이 아래 코드가 원본 코드 (주석 해제 필요)
"""
# 결과 저장할 리스트
min_diff_indices = []


# test_set 각 행에 대해 처리
for _, row in test_set.iterrows():
    # 각 열과의 차이 계산
    diffs = np.abs(grouped_data[['plan_cpu', 'plan_mem']] - row)

    # 각 행별로 가장 작은 값의 인덱스
    min_diff_index = diffs.sum(axis=1).idxmin()

    # 결과 저장
    min_diff_indices.append(min_diff_index)

# {8: 597799, 17: 78576, 7: 25336, 11: 16198, 5: 20556, 3: 116782, 2: 41422, 6: 20973, 21: 8682, 13: 27532,
# 10: 7371, 9: 6606, 14: 12143, 4: 10297, 18: 2123, 16: 1111, 1: 2858, 15: 901, 20: 932, 12: 1109, 19: 238, 0: 207}


cluster = [169, 16, 257, 3, 216, 232, 1295, 245, 145, 409, 202, 115]
count = 0

index_dict = {}
for i in range(0, len(min_diff_indices)):
    if min_diff_indices[i] in index_dict:
        index_dict[min_diff_indices[i]] += 1
    else:
        index_dict[min_diff_indices[i]] = 1


# sorted_weights = dict(sorted(index_dict.items(), key=lambda item: item[1], reverse=True))
# print(sorted_weights)


# total = 0
# for j in index_dict:
#     total = total + index_dict[j]

# 결과 출력
for idx in min_diff_indices:
    if(idx not in cluster):
        count = count+1

print('Graph Learn task resource clustering 정확도: ', 100-(count/len(min_diff_indices)*100), '%')





#각 group별 inst id 개수 시각화
plt.figure(figsize=(8, 5))
plt.bar(range(len(grouped_data)), grouped_data['inst_id_count'])
plt.xlabel('Index')
plt.ylabel('inst_id Count')
plt.title('inst_id Count per Row')

plt.show()





"""===== inst 개수 3D 시각화 ====="""
# 색상 팔레트 설정
palette = sns.color_palette("viridis", as_cmap=True)

# 크기 설정
fig = plt.figure(figsize=(20, 10))

ax = fig.add_subplot(111, projection='3d')

# scatter plot 그리기
scatter = ax.scatter(
    xs=grouped_data_for_3D['plan_cpu'],
    ys=grouped_data_for_3D['plan_mem'],
    zs=grouped_data_for_3D['plan_gpu'],
    c=[len(ids) for ids in grouped_data_for_3D['inst_id']],  # 점의 크기는 inst_id 개수에 따라 다르게 설정
    cmap=palette,
    alpha=0.7,  # 투명도
    edgecolors="w",  # 점의 테두리 색상
    linewidths=0.5,  # 테두리 두께
    s=100
)

# color bar 추가
cbar = plt.colorbar(scatter)
cbar.set_label('inst_id Count', rotation=270, labelpad=15)

# 라벨 및 제목 설정
ax.set_xlabel('Plan CPU')
ax.set_ylabel('Plan MEM')
ax.set_zlabel('Plan GPU')
plt.title('Graph Learn task: Node & Resource Allocation Status')

# 플롯 보이기
plt.show()