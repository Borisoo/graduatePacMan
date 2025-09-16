import math
import os.path
import datetime
import numpy as np
import pandas as pd


class DataProcessor():
    def __init__(self, file_name, index, head, tail, save_dir, seq_length, global_max_dist): 
        # global_max_dist is used to normalize distance values to [0,1] range
        self.df = pd.read_csv(file_name, low_memory=False, header=0) 
        self.temp_df = self.df
        self.temp_df_ndarry = self.temp_df.values
        self.index = index
        self.head = head
        self.tail = tail
        self.save_dir = save_dir
        self.index_no = len(index)
        self.seq_length = seq_length
        self.global_max_dist = global_max_dist

    # 사용할 열 선택
    # def col_selector(self):
    #     self.temp_df = self.temp_df.get(self.index)
    #     return self.temp_df

    # 사용할 열 선택
    def col_selector(self, columns=None):
        if columns is None:
            # columns = self.index
            columns = self.temp_df.columns  # 모든 열 선택
        self.temp_df = self.temp_df.get(columns)
        return self.temp_df

    # 앞뒤 자르기
    def row_cutter(self):
        self.temp_df = self.temp_df.loc[self.head:self.tail, :] #tail is included, 310000 is too big for index
        return self.temp_df

    # 극좌표로 변환하여 반환（n,2）
    def vector_to_pole(self, index1, index2):
        # col1 = self.temp_df.loc[:, index1]
        # col2 = self.temp_df.loc[:, index2]
        col1 = self.temp_df[index1].values.astype(float)
        col2 = self.temp_df[index2].values.astype(float)
        distance = np.sqrt(np.power(col1, 2) + np.power(col2, 2))
        angle = np.arctan2(col2, col1)
        return np.c_[angle, distance]
    
    # 극좌표로 변환하여 반환（n,3), sin, cos, distance, 
    # distance is not normalized, it is better to normalize it to [0,1] range by dividing it by the maximum distance in the dataset
    def vector_to_pole3(self, index1, index2):
        # 원본 x, y 벡터
        col1 = self.temp_df[index1].values.astype(float)
        col2 = self.temp_df[index2].values.astype(float)

        # 거리 계산
        distance = np.sqrt(col1**2 + col2**2)

        # 각도 계산
        angle = np.arctan2(col2, col1)

        # 각도를 sin, cos으로 인코딩
        sin_theta = np.sin(angle)
        cos_theta = np.cos(angle)

        # (sin, cos, distance) 순서로 합쳐서 리턴
        return np.c_[sin_theta, cos_theta, distance]

    def compute_global_max_distance(self):
        # 모든 벡터 조합에서 거리 배열을 모아 최대값을 뽑는다
        vec_cols = [
            ('player_x', 'player_y'),
            ('enemy1_to_player_x', 'enemy1_to_player_y'),
            ('enemy2_to_player_x', 'enemy2_to_player_y'),
            ('goal_to_player_x', 'goal_to_player_y'),
            ('goal_to_des_x', 'goal_to_des_y'),
            ('player_to_des_x', 'player_to_des_y'),
        ]
        all_dists = []
        for xcol, ycol in vec_cols:
            x = self.temp_df[xcol].astype(float).values
            y = self.temp_df[ycol].astype(float).values
            all_dists.append(np.sqrt(x**2 + y**2))
        
        return np.nanmax(np.concatenate(all_dists)) 
        #all_dists is not null, np.max(np.concatenate(all_dists)) is None, i wanna know why, but np.nanmax works well 
    
    def vector_to_pole_norm(self, idx1, idx2):
        x = self.temp_df[idx1].astype(float).values
        y = self.temp_df[idx2].astype(float).values

        # 원거리 계산
        distance = np.sqrt(x**2 + y**2)

        # 1) 벡터별 개별 정규화: 
        # max_dist = distance.max()
        # distance = distance / max_dist

        # 2) 전체 벡터 공통 정규화:
        distance = distance / self.global_max_dist

        # 각도 → sin, cos 인코딩
        angle = np.arctan2(y, x)
        sin_t = np.sin(angle)
        cos_t = np.cos(angle)

        return np.c_[sin_t, cos_t, distance]

    def vectors_to_pole_norm(self):
        parts = [
            self.vector_to_pole_norm('player_x', 'player_y'),
            self.vector_to_pole_norm('enemy1_to_player_x', 'enemy1_to_player_y'),
            self.vector_to_pole_norm('enemy2_to_player_x', 'enemy2_to_player_y'),
            self.vector_to_pole_norm('goal_to_player_x', 'goal_to_player_y'),
            self.vector_to_pole_norm('goal_to_des_x', 'goal_to_des_y'),
            self.vector_to_pole_norm('player_to_des_x', 'player_to_des_y'),
        ]
        poles_np = np.concatenate(parts, axis=1)

        poles_pd = pd.DataFrame(poles_np, columns=['player_x', 'player_y', 'player_d',
                                            'enemy1_to_player_x', 'enemy1_to_player_y', 'enemy1_to_player_d',
                                            'enemy2_to_player_x', 'enemy2_to_player_y', 'enemy2_to_player_d',
                                            'goal_to_player_x', 'goal_to_player_y', 'goal_to_player_d',
                                            'goal_to_des_x', 'goal_to_des_y', 'goal_to_des_d',
                                            'player_to_des_x', 'player_to_des_y', 'player_to_des_d'])
        self.temp_df = pd.concat(
            [self.temp_df.loc[:, 'label':'Timestamp'], poles_pd, self.temp_df.loc[:, 'input_x':'input_all']], axis=1) #there are no label0 in raw data without label, so it will cause error
        return self.temp_df


    # 전체를 극좌표로 변환(12열 -> 24열, 24열 is np.c_[angle,distance]), which is harmful in learning of lstm, 
    # because angle changes discontinuously when object moves around the reference point, for example, 
    # angle changes from -pi to pi suddenly, which makes it difficult for lstm to learn, 
    # and it needs more complex model structure or more neurons to learn, 
    # so the learning efficiency is low and the prediction accuracy is low, so there is no need to convert to polar coordinates. 
    # 그러나 각을 인코딩을하면 괜찮을 수도 있다. 
    def vectors_to_pole(self):
        player = self.vector_to_pole3('player_x', 'player_y') # (n,3), 3 is sin, cos, distance
        enemy1 = self.vector_to_pole3('enemy1_to_player_x', 'enemy1_to_player_y')
        enemy2 = self.vector_to_pole3('enemy2_to_player_x', 'enemy2_to_player_y')
        goal_player = self.vector_to_pole3('goal_to_player_x', 'goal_to_player_y')
        goal_des = self.vector_to_pole3('goal_to_des_x', 'goal_to_des_y')
        player_des = self.vector_to_pole3('player_to_des_x', 'player_to_des_y')
        poles_np = np.concatenate((player, enemy1, enemy2, goal_player, goal_des, player_des), axis=1)
        # poles_pd = pd.DataFrame(poles_np, columns=['player_x', 'player_y',
        #                                            'enemy1_to_player_x', 'enemy1_to_player_y',
        #                                            'enemy2_to_player_x', 'enemy2_to_player_y',
        #                                            'goal_to_player_x', 'goal_to_player_y',
        #                                            'goal_to_des_x', 'goal_to_des_y',
        #                                            'player_to_des_x', 'player_to_des_y'])
        poles_pd = pd.DataFrame(poles_np, columns=['player_x', 'player_y', 'player_d',
                                            'enemy1_to_player_x', 'enemy1_to_player_y', 'enemy1_to_player_d',
                                            'enemy2_to_player_x', 'enemy2_to_player_y', 'enemy2_to_player_d',
                                            'goal_to_player_x', 'goal_to_player_y', 'goal_to_player_d',
                                            'goal_to_des_x', 'goal_to_des_y', 'goal_to_des_d',
                                            'player_to_des_x', 'player_to_des_y', 'player_to_des_d'])
        self.temp_df = pd.concat(
            [self.temp_df.loc[:, 'label':'Timestamp'], poles_pd, self.temp_df.loc[:, 'input_x':'input_all']], axis=1) #there are no label0 in raw data without label, so it will cause error
        return self.temp_df

    # 입력 앞당기기, forward_steps 만큼, (n, m) -> (n-forward_steps, m+forward_steps)
    # 입력 앞당기기는 미래 예측을 위해 사용, forward_steps는 예측하고자 하는 시간 간격
    # 예를 들어 forward_steps가 5이면, 5타임스텝 앞의 입력을 현재 입력으로 사용
    # forward_steps는 seq_length보다 작아야 함
    def input_forward(self, forward_steps):
        row_number = self.temp_df.shape[0]
        
        ####
        # vec_part = self.temp_df.iloc[:, 5:]
        # ['label0','label1', 'No', 'Timestamp', 'player_x', 'player_y', 'enemy1_to_player_x', 'enemy1_to_player_y', 'enemy2_to_player_x', 'enemy2_to_player_y', 'goal_to_player_x', 'goal_to_player_y', 'player_to_des_x', 'player_to_des_y']
        # input_part = self.temp_df.iloc[:, :15]['input_x', 'input_y', 'input_z', 'input_w', 'input_all'] # 맨 앞 15열의 'input_x' ~ 'input_all'
        # right = input_part.loc[forward_steps:row_number, :]
        # left = vec_part.loc[1:row_number - forward_steps, :]
        ####
        
        left = self.temp_df.loc[:, 'Timestamp':'player_to_des_y'] # label0 ~ player_to_des_y, 맨 앞 15열의 'label' ~ 'player_to_des_y', label0 is added when using vectors_to_pole3 
        right = self.temp_df.loc[:, 'input_x':'input_all'] 
        # 맨 앞 15열의 'input_x' ~ 'input_all',  input_all은 1열, input_x, input_y, input_z, input_w는 4열, 총 5열, input_all은 0~4까지의 합

        left = left.loc[0:row_number - forward_steps - 1, :]
        right = right.loc[forward_steps:row_number - 1, :]

        # 인덱스 값 재설정
        left = left.reset_index(drop=True) #drop=True: 기존 인덱스 열을 삭제
        right = right.reset_index(drop=True)

        self.temp_df = pd.concat([left, right], axis=1)
        return self.temp_df

    # 차이값(차분) 검출, 평균값을 deviation 열에 추가, 각 열의 변화량의 평균을 구하는 것
    def deviation_calculator(self):
        # minuend = self.temp_df.loc[:, 'Timestamp':'player_to_des_y']
        minuend = self.temp_df.loc[:, 'Timestamp':'player_to_des_d'] # 'player_to_des_d' is added when using vectors_to_pole3
        subtract = minuend
        # subtract = pd.DataFrame(np.insert(subtract.values, 0, values=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], axis=0),
        #                         columns=['Timestamp', 'player_x', 'player_y', 'enemy1_to_player_x',
        #                                  'enemy1_to_player_y', 'enemy2_to_player_x', 'enemy2_to_player_y',
        #                                  'goal_to_player_x', 'goal_to_player_y', 'goal_to_des_x', 'goal_to_des_y',
        #                                  'player_to_des_x', 'player_to_des_y'])
        
        subtract = pd.DataFrame(np.insert(subtract.values, 0, values=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], axis=0),
                                columns=['Timestamp', 'player_x', 'player_y', 'player_d', 'enemy1_to_player_x',
                                         'enemy1_to_player_y', 'enemy1_to_player_d', 'enemy2_to_player_x', 'enemy2_to_player_y',
                                         'enemy2_to_player_d', 'goal_to_player_x', 'goal_to_player_y', 'goal_to_player_d', 'goal_to_des_x', 'goal_to_des_y',
                                         'goal_to_des_d', 'player_to_des_x', 'player_to_des_y', 'player_to_des_d'])
                
        # minuend와 subtract는 같은 열 구조의 DataFrame
        numeric_cols = minuend.select_dtypes(include=[np.number]).columns
        result = minuend[numeric_cols].sub(subtract[numeric_cols])

        # zero_row = np.zeros((1, minuend.shape[1]))        # subtract: 0행 추가
        # subtract_values = np.vstack([zero_row, minuend.values[:-1]])  # 한 칸씩 밀기
        # subtract = pd.DataFrame(subtract_values, columns=minuend.columns)
        # result = minuend[numeric_cols] - subtract[numeric_cols]        # 숫자형 열만 선택해서 연산
        
        self.temp_df['deviation'] = result.mean(axis=1)
        self.temp_df = pd.concat(
            [self.temp_df.loc[:, 'label':'No'], result, self.temp_df.loc[:, 'input_x':'input_all']], axis=1)
        return self.temp_df

    #  def input_change_detector(self):

    # 저장
    def save(self):
        path = os.path.join(self.save_dir, datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".csv")
        # 2. 디렉터리만 뽑아서 생성
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        self.temp_df.to_csv(path, index=False)
        print('......saved......')


    # 순서대로 데이터 추출
    def next_window(self, i):
        window = self.temp_df_ndarry[i:i + self.seq_length, :]
        
        ####
        # print('win')
        # print(window)
        # print(window.shape)
        ####
        
        # x = window[:, 0:] # 첫 번째 열(레이블) 포함, 입력에 레이블 포함시키는 것은 좋지 않음
        # y = window[0, 0] # 시퀀스의 첫 번째 레이블 사용

        x = window[:, 2:]  # 첫 번째 열(Label, No) 제외
        y = window[-1, 0]  # 시퀀스의 마지막 레이블 사용
    
        ####
        # print('x')
        # print(x)
        # print(x.shape)
        
        # print('y')
        # print(y)
        # print(y.shape)
        ####
        
        return x, y

    # LSTM의 입력 형식, 출력 형식으로 변환, 
    # (n, time_steps, channels),
    # n is number of samples, time_steps is seq_length, channels is number of features, 
    # channels = index_no - 2 (label, No 제외), y는 (n,) 형태로 리턴, label만 사용
    def lstm_input_convertor(self):
        data_x = []
        data_y = []
        self.temp_df_ndarry = self.temp_df.values
        print(self.temp_df_ndarry.shape) #
        length = self.temp_df.shape[0]
        for i in range(0, length - self.seq_length, 100): #100개씩 띄우는 것은 과적합 방지, 데이터 양이 많아지면 200개씩 띄워도 됨
            x_win, y_win = self.next_window(i)
            print(i) #
            data_x.append(x_win)
            data_y.append(y_win)

        return np.array(data_x), np.array(data_y) 


if __name__ == '__main__':
    a = DataProcessor('raw_training_data/record1.csv',
                    index=['Timestamp',
                            'player_x', 'player_y', 'player_d',
                            'enemy1_to_player_x', 'enemy1_to_player_y', 'enemy1_to_player_d',
                            'enemy2_to_player_x', 'enemy2_to_player_y', 'enemy2_to_player_d',
                            'goal_to_player_x', 'goal_to_player_y', 'goal_to_player_d',
                            'goal_to_des_x', 'goal_to_des_y', 'goal_to_des_d',
                            'player_to_des_x', 'player_to_des_y', 'player_to_des_d'],
                    head=10000,
                    tail=310000,
                    save_dir='training_data',
                    seq_length=2000,
                    global_max_dist=1000) # you can set global_max_dist to any value, but it is better to set it to the maximum distance in the dataset, because it will make the distance values in the range of 0 to 1, which is better for learning of lstm

    a.global_max_dist = a.compute_global_max_distance()
    a.vectors_to_pole_norm()
    a.deviation_calculator()
    a.col_selector()
    a.row_cutter()
    a.input_forward(5) # 5타임스텝 앞의 입력을 현재 입력으로 사용
    x, y = a.lstm_input_convertor()
    
    print(x.shape)
    print(y.shape)
    
    ####
    print(a.temp_df)
    b = a.temp_df.loc[:, 'enemy1_to_player_x'] # x is sin(angle) in polar coordinates in case of vector_to_pole3
    c = a.temp_df.loc[:, 'enemy1_to_player_y'] # y is cos(angle) in polar coordinates in case of vector_to_pole3
    bc=  a.temp_df.loc[:, 'enemy1_to_player_d'] # distance in polar coordinates in case of vector_to_pole3, regularization is needed, because distance is too large compared to sin, cos values, which are -1 to 1, so it will cause learning problem in lstm, if you use distance, you need to normalize it like distance/max_distance, where max_distance is the maximum distance in the dataset, or you can just not use distance, because sin, cos are enough to represent the direction, and distance is not that important in this case.
    d = np.sqrt(np.power(b, 2) + np.power(c, 2))
    e = np.arctan2(c, b) # in case of vector_to_pole3, angle = arctan2(y, x), x is sin(angle), y is cos(angle)
    f = pd.DataFrame(np.c_[d, e])

    a.col_selector(['enemy1_to_player_x', 'enemy1_to_player_y', 'enemy1_to_player_d'])
    m = a.vector_to_pole3('enemy1_to_player_x', 'enemy1_to_player_y') # in case of vector_to_pole3, it will return (n,3) array, 3 is sin, cos, distance
    print(m)
    a.col_selector('enemy1_to_player_x')
    print(m)
    n = pd.DataFrame(m) #DataFrame from ndarray, columns=['angle', 'distance']
    print(n)

