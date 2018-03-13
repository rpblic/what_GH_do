from collections import deque
from pprint import *
from numpy import linalg as LA
import numpy as np

users = [
    { "id": 0, "name": "Hero" },
    { "id": 1, "name": "Dunn" },
    { "id": 2, "name": "Sue" },
    { "id": 3, "name": "Chi" },
    { "id": 4, "name": "Thor" },
    { "id": 5, "name": "Clive" },
    { "id": 6, "name": "Hicks" },
    { "id": 7, "name": "Devin" },
    { "id": 8, "name": "Kate" },
    { "id": 9, "name": "Klein" }
]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

# 각 유저마자 "friends"라는 Key에 빈 리스트를 Value로 할당
for user in users:
    user["friends"] = []
    
for i, j in friendships:
    # 각 friendship을 튜플로 갖는 리스트인 friendship에서
    # 방향성 없는 네트워크이므로 양쪽에 서로 친구 추가
    users[i]["friends"].append(users[j]) # j를 i의 친구로 추가
    users[j]["friends"].append(users[i]) # i를 j의 친구로 추가

def degree_centrality(users, user_id):
    numerator = len(users[user_id]["friends"])
    denominator = 0
    for user in users:
        denominator += len(user["friends"])
    return numerator / denominator

def shortest_paths_from(from_user):

    # 특정 사용자로부터 다른 사용자까지의 모든 최단 경로를 포함하는 사전

    shortest_paths_to = {from_user["id"] : [[]]}

    # 확인해야 하는 (이전 사용자, 다음 사용자) 큐
    # 모든 (from_user, from_user의 친구) 쌍으로 시작
    # 각 user에 해당하는 건 사전 자료형에 해당한다는 점에 유의해야 함

    frontier = deque((from_user, friend) for friend in from_user["friends"])

    while frontier:

        prev_user, user = frontier.popleft() # 큐의 첫 번째 사용자를 뽑고 제거
        # prev_user와 user의 쌍
        user_id = user["id"] # 바로 윗 줄에서 빠져나온 from_user의 친구

        # 큐에 사용자를 추가하는 방법을 고려해 보면
        # prev_user까지의 최단 경로를 이미 알고 있을 수도 있다
        # 왜냐하면 각 노드는 중복을 허용하기 때문에 우회하여 가는 길도 가능

        paths_to_prev_user = shortest_paths_to[prev_user["id"]]

        new_paths_to_user = [path + [user_id] for path in paths_to_prev_user]

        old_paths_to_user = shortest_paths_to.get(user_id, [])
        # 사전 자료형의 메소드인 get은 key를 파라미터로 해당 value를 반환
        # 만약 해당 key가 없는 경우에는 파라미터를 하나 더 추가해
        # 그 내용이 반환되도록 하여 에러 배제

        if old_paths_to_user:
            old_min_path_length = len(min(old_paths_to_user, key = lambda x : len(x)))
        else:
            old_min_path_length = float("inf")

        # 길지 않은 새로운 경로만 저장

        new_min_path_length = len(min(new_paths_to_user, key = lambda x : len(x)))
        
        if new_min_path_length < old_min_path_length:
            shortest_paths_to[user_id] = [path for path in new_paths_to_user \
                                          if len(path) == new_min_path_length]

        elif new_min_path_length == old_min_path_length:
            new_paths_to_user = [path for path in new_paths_to_user \
                                 if len(path) == new_min_path_length \
                                 and path not in old_paths_to_user]
            
            shortest_paths_to[user_id] = old_paths_to_user + new_paths_to_user

        else:
            pass

        # 처음 나오는 이웃을 frontier에 추가

        frontier.extend((user, friend) for friend in user["friends"] \
                        if friend["id"] not in shortest_paths_to)

        # deque 클래스의 메소드인 extend는 해당 deque의 뒤에 파라미터들이 줄을 서도록 함

    return shortest_paths_to

for user in users:
    user["shortest_paths"] = shortest_paths_from(user)

for user in users:
    user["betweenness_centrality"] = 0.0

for source in users:
    source_id = source["id"]
    for target_id, paths in source["shortest_paths"].items():
        if source_id < target_id: # 중복되지 않게 (아래의 알고리즘을 확인하면 
            num_paths = len(paths) # 총 최단 경로 개수
            contrib = 1 / num_paths # 기여를 할 때마다 betweenness_centrality에 더해지는 1 / 총 최단 경로 개수 값
            for path in paths: # 각 최단 경로에 대해
                for id in path: # 포함되는 각 노드 중
                    if id not in [source_id, target_id]: # 출발 노드와 도착 노드 모두 아닌 것에 한해
                        users[id]["betweenness_centrality"] += contrib # 해당 유저의 매개 중심성에 1 / 총 최단 경로 개수 값

def farness(user): # 해당 유저가 갖는, 타 노드 각각과의 최단 경로들의 길이 합
    return sum(len(paths[0]) \
               for paths in user["shortest_paths"].values()) # 고립 노드의 경우에는 farness가 과소평가 될 우려 있음

for user in users:
    user["closeness_centrality"] = 1 / farness(user)

adj_matrix = np.zeros((len(users), len(users))) # (유저 수)차 정사각 영행렬을 정의

for i, j in friendships: # 서로 연결된 경우 1을 할당하여 인접행렬로
    adj_matrix[i][j] = 1
    adj_matrix[j][i] = 1

w, v = LA.eig(adj_matrix) # numpy.linalg.eig(행렬)은 eigenvalue의 array와 eigenvector의 array를 반환

for user_id, eigenvector_centrality in enumerate(-v[:, 0]):
    print(user_id, eigenvector_centrality, sep = " : ")
# 0번 인덱스에 해당하는 eigenvalue에 대응하는 eigenvector의 성분을 모두 양의 실수로 변환하고 인덱스 순서대로 출력
