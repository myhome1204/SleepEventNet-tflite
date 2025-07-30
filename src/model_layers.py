import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable



@register_keras_serializable()
class YamnetWrapper(tf.keras.layers.Layer):
    def __init__(self, yamnet_model_path, **kwargs):
        super(YamnetWrapper, self).__init__(**kwargs)
        self.yamnet_model_path = yamnet_model_path
        self.yamnet = tf.saved_model.load(yamnet_model_path)

    def call(self, inputs):
        inputs = tf.reshape(inputs, [-1])  # (1, T) → (T,)
        scores, embeddings, spectrogram = self.yamnet(inputs)
        return scores, embeddings, spectrogram

    def get_config(self):
        config = super().get_config()
        config.update({
            "yamnet_model_path": self.yamnet_model_path
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
class BruxismFrameJudgementLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BruxismFrameJudgementLayer, self).__init__(**kwargs)
        self.bruxism_combination_count  = tf.constant(13,dtype=tf.int32)
        self.bruxism_combination_group_size = tf.constant(3,dtype=tf.int32)
        # self.bruxism_audio_group_size = tf.constant(4,dtype=tf.int32)
        self.bruxism_combination_list = tf.constant(
            [
              # 동물 소리 포함 이갈이
              [(500, 0.1, 0.8), (41, 0.05, 0.5),(0, 0.0, 1.0)],  # 41 Snort
              [(369, 0.1, 0.5), (500, 0.1, 0.5),(0, 0.0, 1.0)],  # 369 Toothbrush
              [(412, 0.05, 0.4), (431, 0.05, 0.4), (470, 0.05, 0.4)],  # 412 : Tools , 431: Wood , 470: Rub
              [(127, 0.1, 0.8), (67, 0.1, 0.7), (103, 0.1, 0.6)],  # Frog, Animal, Wild animals
              [(399, 0.1, 0.9), (403, 0.1, 0.8), (412, 0.1, 0.7)],  # Ratchet, pawl, Gears, Tools
              [(410, 0.1, 0.9), (411, 0.1, 0.8), (398, 0.1, 0.7)],  # Camera, Single-lens reflex camera, Mechanisms
              [(412, 0.1, 0.8), (410, 0.1, 0.7), (398, 0.1, 0.6)],  # Tools, Camera, Mechanisms
              [(435, 0.1, 0.9), (438, 0.1, 0.8), (449, 0.1, 0.7)],  # Glass, Liquid, Stir
              [(374, 0.1, 0.9), (436, 0.1, 0.8), (435, 0.1, 0.7)],  # Coin dropping, Chink, clink, Glass
              [(372, 0.1, 0.9), (434, 0.1, 0.8), (469, 0.1, 0.7)],  # Zipper, Crack, Scrape
              [(500, 0.1, 0.8), (439, 0.1, 0.7), (50, 0.1, 0.6)],  # Inside small room, Splash, Biting
              [(399, 0.1, 0.9), (403, 0.1, 0.8), (410, 0.1, 0.7)],  # Ratchet, pawl, Gears, Camera
              [(410, 0.1, 0.9), (398, 0.1, 0.8), (435, 0.1, 0.7)]  # Camera, Mechanisms, Glass
          ]
         , dtype=tf.float32
        )
        self.bruxism_score = tf.constant(
        [
           2.94, 1.76, 1.76,  1.47,  0.29,
          0.88, 0.01,  0.29,  0.29, 0.01,
           0.29, 0.01, 0.01
        ]
            , dtype=tf.float32
        )
    #여기서 받는 score는 한개의 frame에 대한 결과값임. 즉 (521,)형태
    def call(self,score):
        # print(f"score shape : {score.shape} ")
        score= tf.squeeze(score, axis=0)
        i = tf.constant(0)
        judgement = tf.constant(False)
        def cond(i,judgement):
            return tf.logical_and(i<self.bruxism_combination_count,tf.logical_not(judgement))
        def body(i,judgement):
            combination = tf.gather(self.bruxism_combination_list,i)
            one_combination_judgement = self.one_frame_judgement(combination,score)
            return tf.add(i,1), one_combination_judgement
        index,judgement = tf.while_loop(cond,body,[i,judgement])
        #tf.cond는 파라미터(2번째,3번쨰)에 함수를 넣어줘야함
        judgement = tf.cond(
                    judgement,  # 조건: 현재 judgement가 True일 때만 아래 함수 실행
                    lambda: self.is_non_bruxism_top_10(score),  # True일 때 실행할 함수
                    lambda: tf.constant(False)  # False일 땐 그냥 False 유지
                )

        return tf.cond(
                      judgement,
                      lambda: [judgement,tf.round(tf.gather(self.bruxism_score,index - 1) * 100) / 100],
                      lambda: [judgement,tf.constant(0.0, dtype=tf.float32)]
                  )

    def one_frame_judgement(self,combination,score):
        i = tf.constant(0)
        judgement = tf.constant(True)
        def cond(i,judgement):
          return tf.logical_and(i<self.bruxism_combination_group_size,judgement)
        def body(i,judgement):
          row = combination[i]
          class_number = tf.gather(row,0)
          class_number = tf.cast(class_number, tf.int32)
          min_value = tf.gather(row,1)
          max_value = tf.gather(row,2)
          in_range = tf.logical_and(tf.greater_equal(score[class_number],min_value),tf.less_equal(score[class_number],max_value))
          # tf.print("class:", class_number, " score:",score[class_number], " min:", min_value, " max:", max_value, " → in_range:", in_range)
          judgement = tf.logical_and(judgement,in_range)
          return tf.add(i,1), judgement
        _, final_judgement = tf.while_loop(cond, body, [i, judgement])
        return final_judgement
    #상위 10개에 기침 ,스피치 , 코골이가 있는가
    def is_non_bruxism_top_10(self, score):
    # top 10 인덱스를 정수형으로 가져오기
        top_k_indices = tf.math.top_k(score, 10).indices
        top_k_indices = tf.cast(top_k_indices, dtype=tf.int32)

        # 비교 대상 클래스들
        non_bruxism_classes = tf.constant([38, 42, 0], dtype=tf.int32)

        # broadcasting을 활용하여 비교
        # shape: (10, 3) -> 각 top10 인덱스가 non_bruxism_classes와 같은지 체크
        comparison = tf.equal(tf.expand_dims(top_k_indices, axis=1), non_bruxism_classes)

        # 하나라도 일치하면 True
        has_non_bruxism = tf.reduce_any(comparison)

        # 일치하는 게 없으면 비브럭시즘(True)로 판단
        return tf.logical_not(has_non_bruxism)

    def build(self, input_shape):
        # print("input shape to this layer:", input_shape)
        self.output_dim = input_shape[-1]
        super().build(input_shape)


class BruxismClassificationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BruxismClassificationLayer, self).__init__(**kwargs)
        self.bruxism_frame_judgement_layer = BruxismFrameJudgementLayer()
        self.bruxism_group_size = tf.constant(4,dtype=tf.int32)
        self.bruxism_min_valid_count = tf.constant(2,dtype=tf.int32)
        self.bruxism_required_true_count = tf.constant(2,dtype=tf.int32)
        # 이부분은 그룹사이즈 layer추가되면 parameter로 받아야함
        self.total_score_size = tf.constant(42,dtype=tf.int32)
        self.group_range  = self.total_score_size-self.bruxism_group_size+1

    def call(self,bruxism_result):
        # print("bruxism_result size : ",bruxism_result.shape)
        # bruxism_result= tf.squeeze(bruxism_result, axis=0)
        # print("bruxism_result size : ",bruxism_result.shape)
        frame_judgements,frame_scores = self.convert_all_frames(bruxism_result)
        # tf.print(frame_judgements, summarize=-1)
        # tf.print(frame_scores, summarize=-1)

        i = tf.constant(0)
        judgements_tensor = tf.zeros([self.group_range], dtype=tf.bool)
        scores_tensor = tf.zeros([self.group_range], dtype=tf.float32)
        def cond(i,judgements_tensor,scores_tensor):
            return tf.less(i,self.group_range)
        def body(i,judgements_tensor,scores_tensor):
            group_judgements = frame_judgements[i:i+self.bruxism_group_size]
            group_scores = frame_scores[i:i+self.bruxism_group_size]
            judgement,score= self.group_judgement(group_judgements,group_scores)
            judgements_tensor = tf.tensor_scatter_nd_update(judgements_tensor, [[i]], [judgement])
            scores_tensor = tf.tensor_scatter_nd_update(scores_tensor, [[i]], [score])
            return tf.add(i,1) , judgements_tensor,scores_tensor
        _,judgements_tensor,scores_tensor = tf.while_loop(cond,body,[i,judgements_tensor,scores_tensor])

        true_values = tf.boolean_mask(scores_tensor, judgements_tensor)
        true_count = tf.shape(true_values)[0]
        return tf.cond(
            true_count >= self.bruxism_required_true_count,
            lambda : [tf.constant(True,tf.bool),self.get_top_values(true_values)],
            lambda : [tf.constant(False,tf.bool),tf.constant(0.0,tf.float32)]
        )

    # 그룹 판단함수.
    def group_judgement(self,group_judgement,group_score):
        # 각행의 첫번째 값들 만 빼서 확인
        # tf.print("result : ",result.shape)
        # bool_values = tf.cast(result[:,0],tf.bool)
        # score_values = tf.cast(result[:,1],tf.float32)
        # true인 값의 score만 뽑아 tensor리스트로 저장.

        true_values = tf.boolean_mask(group_score, group_judgement)

        #true_values 의갯수가 곧 true count값임.
        true_count = tf.shape(true_values)[0]
        return tf.cond(
            true_count >= self.bruxism_min_valid_count,
            lambda : [tf.constant(True,tf.bool),tf.reduce_sum(true_values)],
            lambda : [tf.constant(False,tf.bool),tf.constant(0.0,tf.float32)]
        )
    #상위권 group 의 score 값 더해주는함수 self.bruxism_required_true_count 개 만큼
    def get_top_values(self,true_values):
        top_k = tf.math.top_k(true_values, k=self.bruxism_required_true_count)
        result = tf.reduce_sum(top_k.values)
        return tf.cast(result,tf.float32)

    # 프레임수,521 데이터를 한프레임씩 돌려서 프레임수,2 결과로 바꿔주는 함수 ,
    def convert_all_frames(self, bruxism_result):
        i = tf.constant(0)
        total_frame_count = tf.shape(bruxism_result)[0]
        judgements_tensor = tf.zeros([total_frame_count], dtype=tf.bool)
        scores_tensor = tf.zeros([total_frame_count], dtype=tf.float32)

        def cond(i, judgements_tensor,scores_tensor):
            return tf.less(i, total_frame_count)

        def body(i, judgements_tensor,scores_tensor):
            one_frame_judgement, one_frame_score = self.bruxism_frame_judgement_layer(tf.expand_dims(bruxism_result[i], axis=0))
            judgements_tensor = tf.tensor_scatter_nd_update(judgements_tensor, [[i]], [one_frame_judgement])
            scores_tensor = tf.tensor_scatter_nd_update(scores_tensor, [[i]], [one_frame_score])
            return tf.add(i, 1), judgements_tensor,scores_tensor

        _, judgements_tensor,scores_tensor = tf.while_loop(cond, body, [i,judgements_tensor,scores_tensor])

        return judgements_tensor,scores_tensor


#결과값 판단 ,점수
class SnoringFrameJudgementLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SnoringFrameJudgementLayer, self).__init__(**kwargs)
        self.snoring_combination_count  = tf.constant(3,dtype=tf.int32)
        self.snoring_combination_group_size = tf.constant(3,dtype=tf.int32)
        # self.bruxism_audio_group_size = tf.constant(4,dtype=tf.int32)
        self.snoring_combination_list = tf.constant(
            [
                [[38, 0.8, 1.0], [36, 0.2, 0.6],[0, 0.0, 1.0]],
                [[38, 0.9, 1.0], [23, 0.2, 0.8], [36, 0.2, 0.5]],
                [[38, 0.8, 1.0], [0, 0.0, 0.5],[0, 0.0, 1.0]
              ]
            ]
            , dtype=tf.float32
        )
        self.snoring_score = tf.constant(
        [
           8.67, 0.01, 1.33,
        ]
            , dtype=tf.float32
        )


    #여기서 받는 score는 한개의 frame에 대한 결과값임. 즉 (521,)형태
    def call(self,score):
        score= tf.squeeze(score, axis=0)
        i = tf.constant(0)
        judgement = tf.constant(False)
        def cond(i,judgement):
            return tf.logical_and(i<self.snoring_combination_count,tf.logical_not(judgement))
        def body(i,judgement):
            combination = tf.gather(self.snoring_combination_list,i)
            one_combination_judgement = self.one_frame_judgement(combination,score)
            return tf.add(i,1), one_combination_judgement
        index , judgement = tf.while_loop(cond,body,[i,judgement])
        #tf.cond는 파라미터(2번째,3번쨰)에 함수를 넣어줘야함
        return tf.cond(
                      judgement,
                      lambda: [judgement,tf.round(tf.gather(self.snoring_score,index - 1) * 100) / 100],
                      lambda: [judgement,0.0]
                  )

    def one_frame_judgement(self,combination,score):
        i = tf.constant(0)
        judgement = tf.constant(True)
        def cond(i,judgement):
          return tf.logical_and(i<self.snoring_combination_group_size,judgement)
        def body(i,judgement):
          row = combination[i]
          class_number = tf.gather(row,0)
          class_number = tf.cast(class_number, tf.int32)
          min_value = tf.gather(row,1)
          max_value = tf.gather(row,2)
          in_range = tf.logical_and(tf.greater_equal(score[class_number],min_value),tf.less_equal(score[class_number],max_value))
          judgement = tf.logical_and(judgement,in_range)
          return tf.add(i,1), judgement
        _, final_judgement = tf.while_loop(cond, body, [i, judgement])
        return final_judgement



class SnoringClassificationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SnoringClassificationLayer, self).__init__(**kwargs)
        self.snoring_frame_judgement_layer = SnoringFrameJudgementLayer()
        self.snoring_group_size = tf.constant(5,dtype=tf.int32)
        self.snoring_min_valid_count = tf.constant(3,dtype=tf.int32)
        self.snoring_required_true_count = tf.constant(2,dtype=tf.int32)
        # 이부분은 그룹사이즈 layer추가되면 parameter로 받아야함
        self.total_score_size = tf.constant(42,dtype=tf.int32)
        self.group_range  = self.total_score_size-self.snoring_group_size+1

    def call(self,snoring_result):

        # snoring_result= tf.squeeze(snoring_result, axis=0)
        frame_judgements,frame_scores = self.convert_all_frames(snoring_result)
        # tf.print(frame_judgements, summarize=-1)
        # tf.print(frame_scores, summarize=-1)
        # tf.print(frame_scores, summarize=-1)
        i = tf.constant(0)
        judgements_tensor = tf.zeros([self.group_range], dtype=tf.bool)
        scores_tensor = tf.zeros([self.group_range], dtype=tf.float32)
        def cond(i,judgements_tensor,scores_tensor):
            return tf.less(i,self.group_range)
        def body(i,judgements_tensor,scores_tensor):
            group_judgements = frame_judgements[i:i+self.snoring_group_size]
            group_scores = frame_scores[i:i+self.snoring_group_size]
            judgement,score= self.group_judgement(group_judgements,group_scores)

            judgements_tensor = tf.tensor_scatter_nd_update(judgements_tensor, [[i]], [judgement])
            scores_tensor = tf.tensor_scatter_nd_update(scores_tensor, [[i]], [score])

            return tf.add(i,1) , judgements_tensor,scores_tensor
        _,judgements_tensor,scores_tensor = tf.while_loop(cond,body,[i,judgements_tensor,scores_tensor])
        # tf.print(judgements_tensor, summarize=-1)
        # tf.print(scores_tensor, summarize=-1)
        true_values = tf.boolean_mask(scores_tensor, judgements_tensor)
        true_count = tf.shape(true_values)[0]
        # tf.print("true_count :",true_count)
        # tf.print("true_values :",true_values)

        return tf.cond(
            true_count >= self.snoring_required_true_count,
            lambda : [tf.constant(True,tf.bool),self.get_top_values(true_values)],
            lambda : [tf.constant(False,tf.bool),tf.constant(0.0,tf.float32)]
        )

    # 그룹 판단함수.
    def group_judgement(self,group_judgement,group_score):

        true_values = tf.boolean_mask(group_score, group_judgement)

        #true_values 의갯수가 곧 true count값임.
        true_count = tf.shape(true_values)[0]
        return tf.cond(
            true_count >= self.snoring_min_valid_count,
            lambda : [tf.constant(True,tf.bool),tf.reduce_sum(true_values)],
            lambda : [tf.constant(False,tf.bool),tf.constant(0.0,tf.float32)]
        )
    #상위권 group 의 score 값 더해주는함수 self.bruxism_required_true_count 개 만큼
    def get_top_values(self,true_values):
        top_k = tf.math.top_k(true_values, k=self.snoring_required_true_count)
        result = tf.reduce_sum(top_k.values)
        return tf.cast(result,tf.float32)

    # 프레임수,521 데이터를 한프레임씩 돌려서 프레임수,2 결과로 바꿔주는 함수 ,
    def convert_all_frames(self, snoring_result):
        i = tf.constant(0)
        total_frame_count = tf.shape(snoring_result)[0]
        judgements_tensor = tf.zeros([total_frame_count], dtype=tf.bool)
        scores_tensor = tf.zeros([total_frame_count], dtype=tf.float32)

        def cond(i, judgements_tensor,scores_tensor):
            return tf.less(i, total_frame_count)

        def body(i, judgements_tensor,scores_tensor):
            one_frame_judgement, one_frame_score = self.snoring_frame_judgement_layer(tf.expand_dims(snoring_result[i], axis=0))
            judgements_tensor = tf.tensor_scatter_nd_update(judgements_tensor, [[i]], [one_frame_judgement])
            scores_tensor = tf.tensor_scatter_nd_update(scores_tensor, [[i]], [one_frame_score])
            return tf.add(i, 1), judgements_tensor,scores_tensor

        _, judgements_tensor,scores_tensor = tf.while_loop(cond, body, [i,judgements_tensor,scores_tensor])

        return judgements_tensor,scores_tensor


#결과값 : 판단,점수
class SpeechFrameJudgementLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpeechFrameJudgementLayer, self).__init__(**kwargs)
        self.speech_combination_count  = tf.constant(6,dtype=tf.int32)
        self.speech_combination_group_size = tf.constant(2,dtype=tf.int32)
        # self.bruxism_audio_group_size = tf.constant(4,dtype=tf.int32)
        self.speech_combination_list = tf.constant(
            [
              [[0, 0.6, 1.0],[0, 0.0, 1.0]],
              [[0, 0.5, 0.7], [2, 0.3, 0.7]],
              [[0, 0.5, 0.7], [3, 0.2, 0.5]],
              [[0, 0.5, 0.7], [5, 0.2, 0.4]],
              [[0, 0.5, 0.7], [132, 0.2, 0.5]],
              [[0, 0.5, 0.7], [36, 0.1, 0.3]]
            ], dtype=tf.float32
        )
        self.speech_score = tf.constant(

         [5, 1.0, 1.0,  1.5, 1.0,  1.2]
            , dtype=tf.float32
        )


    #여기서 받는 score는 한개의 frame에 대한 결과값임. 즉 (521,)형태
    def call(self,score):
        score= tf.squeeze(score, axis=0)
        i = tf.constant(0)
        judgement = tf.constant(False)
        def cond(i,judgement):
            return tf.logical_and(i<self.speech_combination_count,tf.logical_not(judgement))
        def body(i,judgement):
            combination = tf.gather(self.speech_combination_list,i)
            one_combination_judgement = self.one_frame_judgement(combination,score)
            return tf.add(i,1), one_combination_judgement
        index , judgement = tf.while_loop(cond,body,[i,judgement])
        #tf.cond는 파라미터(2번째,3번쨰)에 함수를 넣어줘야함
        return tf.cond(
                      judgement,
                      lambda: [judgement,tf.round(tf.gather(self.speech_score,index - 1) * 100) / 100],
                      lambda: [judgement,0.0]
                  )

    def one_frame_judgement(self,combination,score):
        i = tf.constant(0)
        judgement = tf.constant(True)
        def cond(i,judgement):
          return tf.logical_and(i<self.speech_combination_group_size,judgement)
        def body(i,judgement):
          row = combination[i]
          class_number = tf.gather(row,0)
          class_number = tf.cast(class_number, tf.int32)
          min_value = tf.gather(row,1)
          max_value = tf.gather(row,2)
          in_range = tf.logical_and(tf.greater_equal(score[class_number],min_value),tf.less_equal(score[class_number],max_value))
          judgement = tf.logical_and(judgement,in_range)
          return tf.add(i,1), judgement
        _, final_judgement = tf.while_loop(cond, body, [i, judgement])
        return final_judgement



class SpeechClassificationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpeechClassificationLayer, self).__init__(**kwargs)
        self.speech_frame_judgement_layer = SpeechFrameJudgementLayer()
        self.speech_group_size = tf.constant(5,dtype=tf.int32)
        self.speech_min_valid_count = tf.constant(3,dtype=tf.int32)
        self.speech_required_true_count = tf.constant(2,dtype=tf.int32)
        # 이부분은 그룹사이즈 layer추가되면 parameter로 받아야함
        self.total_score_size = tf.constant(42,dtype=tf.int32)
        self.group_range  = self.total_score_size-self.speech_group_size+1

    def call(self,speech_result):

        # speech_result= tf.squeeze(speech_result, axis=0)
        frame_judgements,frame_scores = self.convert_all_frames(speech_result)
        # tf.print(frame_judgements, summarize=-1)
        # tf.print(frame_scores, summarize=-1)

        i = tf.constant(0)
        judgements_tensor = tf.zeros([self.group_range], dtype=tf.bool)
        scores_tensor = tf.zeros([self.group_range], dtype=tf.float32)
        def cond(i,judgements_tensor,scores_tensor):
            return tf.less(i,self.group_range)
        def body(i,judgements_tensor,scores_tensor):
            group_judgements = frame_judgements[i:i+self.speech_group_size]
            group_scores = frame_scores[i:i+self.speech_group_size]
            judgement,score= self.group_judgement(group_judgements,group_scores)
            judgements_tensor = tf.tensor_scatter_nd_update(judgements_tensor, [[i]], [judgement])
            scores_tensor = tf.tensor_scatter_nd_update(scores_tensor, [[i]], [score])
            return tf.add(i,1) , judgements_tensor,scores_tensor
        _,judgements_tensor,scores_tensor = tf.while_loop(cond,body,[i,judgements_tensor,scores_tensor])

        true_values = tf.boolean_mask(scores_tensor, judgements_tensor)
        true_count = tf.shape(true_values)[0]
        return tf.cond(
            true_count >= self.speech_required_true_count,
            lambda : [tf.constant(True,tf.bool),self.get_top_values(true_values)],
            lambda : [tf.constant(False,tf.bool),tf.constant(0.0,tf.float32)]
        )

    # 그룹 판단함수.
    def group_judgement(self,group_judgement,group_score):
        # 각행의 첫번째 값들 만 빼서 확인
        # tf.print("result : ",result.shape)
        # bool_values = tf.cast(result[:,0],tf.bool)
        # score_values = tf.cast(result[:,1],tf.float32)
        # true인 값의 score만 뽑아 tensor리스트로 저장.

        true_values = tf.boolean_mask(group_score, group_judgement)

        #true_values 의갯수가 곧 true count값임.
        true_count = tf.shape(true_values)[0]
        return tf.cond(
            true_count >= self.speech_min_valid_count,
            lambda : [tf.constant(True,tf.bool),tf.reduce_sum(true_values)],
            lambda : [tf.constant(False,tf.bool),tf.constant(0.0,tf.float32)]
        )
    #상위권 group 의 score 값 더해주는함수 self.bruxism_required_true_count 개 만큼
    def get_top_values(self,true_values):
        top_k = tf.math.top_k(true_values, k=self.speech_required_true_count)
        result = tf.reduce_sum(top_k.values)
        return tf.cast(result,tf.float32)

    # 프레임수,521 데이터를 한프레임씩 돌려서 프레임수,2 결과로 바꿔주는 함수 ,
    def convert_all_frames(self, speech_result):
        i = tf.constant(0)
        total_frame_count = tf.shape(speech_result)[0]
        judgements_tensor = tf.zeros([total_frame_count], dtype=tf.bool)
        scores_tensor = tf.zeros([total_frame_count], dtype=tf.float32)

        def cond(i, judgements_tensor,scores_tensor):
            return tf.less(i, total_frame_count)

        def body(i, judgements_tensor,scores_tensor):
            one_frame_judgement, one_frame_score = self.speech_frame_judgement_layer(tf.expand_dims(speech_result[i], axis=0))
            judgements_tensor = tf.tensor_scatter_nd_update(judgements_tensor, [[i]], [one_frame_judgement])
            scores_tensor = tf.tensor_scatter_nd_update(scores_tensor, [[i]], [one_frame_score])
            return tf.add(i, 1), judgements_tensor,scores_tensor

        _, judgements_tensor,scores_tensor = tf.while_loop(cond, body, [i,judgements_tensor,scores_tensor])

        return judgements_tensor,scores_tensor


#결과값 : 판단,점수
class CoughFrameJudgementLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CoughFrameJudgementLayer, self).__init__(**kwargs)
        self.cough_combination_count  = tf.constant(1,dtype=tf.int32)
        self.cough_combination_group_size = tf.constant(1,dtype=tf.int32)
        # self.bruxism_audio_group_size = tf.constant(4,dtype=tf.int32)
        self.cough_combination_list = tf.constant(

        [
          [[42, 0.1, 1.0]]
        ]
            , dtype=tf.float32
        )
        self.cough_score = tf.constant(
        [
          1.5
        ]
            , dtype=tf.float32
        )

    #여기서 받는 score는 한개의 frame에 대한 결과값임. 즉 (521,)형태
    def call(self,score):
        score= tf.squeeze(score, axis=0)
        i = tf.constant(0)
        judgement = tf.constant(False)
        def cond(i,judgement):
            return tf.logical_and(i<self.cough_combination_count,tf.logical_not(judgement))
        def body(i,judgement):
            combination = tf.gather(self.cough_combination_list,i)
            one_combination_judgement = self.one_frame_judgement(combination,score)
            return tf.add(i,1), one_combination_judgement
        index , judgement = tf.while_loop(cond,body,[i,judgement])
        #tf.cond는 파라미터(2번째,3번쨰)에 함수를 넣어줘야함
        judgement, point = tf.cond(
                    tf.logical_not(judgement),  # 만약 클래스조합에 안걸리더라도 Top10에있는지 확인. 즉 Judgement가 False일떄
                    lambda: self.is_cough_top_10(score),  # 클래스조합에서는 찾기실패했지만 Top_10에 있는지 확인해보기
                    lambda: [judgement,tf.round(tf.gather(self.cough_score,index - 1) * 100) / 100]#judgment가 True였다는거니까 클래스조합에서 점수찾기.
                )

        return judgement,point

    def one_frame_judgement(self,combination,score):
        i = tf.constant(0)
        judgement = tf.constant(True)
        def cond(i,judgement):
          return tf.logical_and(i<self.cough_combination_group_size,judgement)
        def body(i,judgement):
          row = combination[i]
          class_number = tf.gather(row, 0)
          class_number = tf.cast(class_number, tf.int32)
          min_value = tf.gather(row,1)
          max_value = tf.gather(row,2)
          in_range = tf.logical_and(tf.greater_equal(score[class_number],min_value),tf.less_equal(score[class_number],max_value))
          judgement = tf.logical_and(judgement,in_range)
          return tf.add(i,1), judgement
        _, final_judgement = tf.while_loop(cond, body, [i, judgement])
        return final_judgement
    #상위 10개에 기침이 있는가
    def is_cough_top_10(self, score):
        sort_score = tf.math.top_k(score, 10).indices
        sort_score = tf.cast(sort_score, dtype=tf.int32)
        cough_class = tf.constant(42, dtype=tf.int32)

        # cough_class 가 sort_score 안에 존재하는지 확인
        equal_to_cough = tf.equal(sort_score, cough_class)
        has_cough = tf.reduce_any(equal_to_cough)

        return tf.cond(
            has_cough,
            lambda: [tf.constant(True), 1.0],
            lambda: [tf.constant(False), 0.0]
        )


class CoughClassificationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CoughClassificationLayer, self).__init__(**kwargs)
        self.cough_frame_judgement_layer = CoughFrameJudgementLayer()
        self.cough_group_size = tf.constant(3,dtype=tf.int32)
        self.cough_min_valid_count = tf.constant(2,dtype=tf.int32)
        self.cough_required_true_count = tf.constant(2,dtype=tf.int32)
        # 이부분은 그룹사이즈 layer추가되면 parameter로 받아야함
        self.total_score_size = tf.constant(42,dtype=tf.int32)
        self.group_range  = self.total_score_size-self.cough_group_size+1

    def call(self,cough_result):

        # cough_result= tf.squeeze(cough_result, axis=0)
        frame_judgements,frame_scores = self.convert_all_frames(cough_result)
        # tf.print(frame_judgements, summarize=-1)
        # tf.print(frame_scores, summarize=-1)

        i = tf.constant(0)

        judgements_tensor = tf.zeros([self.group_range], dtype=tf.bool)
        scores_tensor = tf.zeros([self.group_range], dtype=tf.float32)

        def cond(i,judgements_tensor,scores_tensor):
            return tf.less(i,self.group_range)
        def body(i,judgements_tensor,scores_tensor):
            group_judgements = frame_judgements[i:i+self.cough_group_size]
            group_scores = frame_scores[i:i+self.cough_group_size]
            judgement,score= self.group_judgement(group_judgements,group_scores)

            judgements_tensor = tf.tensor_scatter_nd_update(judgements_tensor, [[i]], [judgement])
            scores_tensor = tf.tensor_scatter_nd_update(scores_tensor, [[i]], [score])

            return tf.add(i,1) , judgements_tensor,scores_tensor
        _,judgements_tensor,scores_tensor = tf.while_loop(cond,body,[i,judgements_tensor,scores_tensor])

        true_values = tf.boolean_mask(scores_tensor, judgements_tensor)
        true_count = tf.shape(true_values)[0]
        return tf.cond(
            true_count >= self.cough_required_true_count,
            lambda : [tf.constant(True,tf.bool),self.get_top_values(true_values)],
            lambda : [tf.constant(False,tf.bool),tf.constant(0.0,tf.float32)]
        )

    # 그룹 판단함수.
    def group_judgement(self,group_judgement,group_score):
        # 각행의 첫번째 값들 만 빼서 확인
        # tf.print("result : ",result.shape)
        # bool_values = tf.cast(result[:,0],tf.bool)
        # score_values = tf.cast(result[:,1],tf.float32)
        # true인 값의 score만 뽑아 tensor리스트로 저장.

        true_values = tf.boolean_mask(group_score, group_judgement)

        #true_values 의갯수가 곧 true count값임.
        true_count = tf.shape(true_values)[0]
        return tf.cond(
            true_count >= self.cough_min_valid_count,
            lambda : [tf.constant(True,tf.bool),tf.reduce_sum(true_values)],
            lambda : [tf.constant(False,tf.bool),tf.constant(0.0,tf.float32)]
        )
    #상위권 group 의 score 값 더해주는함수 self.bruxism_required_true_count 개 만큼
    def get_top_values(self,true_values):
        top_k = tf.math.top_k(true_values, k=self.cough_required_true_count)
        result = tf.reduce_sum(top_k.values)
        return tf.cast(result,tf.float32)

    # 프레임수,521 데이터를 한프레임씩 돌려서 프레임수,2 결과로 바꿔주는 함수 ,
    def convert_all_frames(self, cough_result):
        i = tf.constant(0)
        total_frame_count = tf.shape(cough_result)[0]

        judgements_tensor = tf.zeros([total_frame_count], dtype=tf.bool)
        scores_tensor = tf.zeros([total_frame_count], dtype=tf.float32)

        def cond(i, judgements_tensor,scores_tensor):
            return tf.less(i, total_frame_count)

        def body(i, judgements_tensor,scores_tensor):
            one_frame_judgement, one_frame_score = self.cough_frame_judgement_layer(tf.expand_dims(cough_result[i], axis=0))

            judgements_tensor = tf.tensor_scatter_nd_update(judgements_tensor, [[i]], [one_frame_judgement])
            scores_tensor = tf.tensor_scatter_nd_update(scores_tensor, [[i]], [one_frame_score])
            return tf.add(i, 1), judgements_tensor,scores_tensor

        _, judgements_tensor,scores_tensor = tf.while_loop(cond, body, [i,judgements_tensor,scores_tensor])

        return judgements_tensor,scores_tensor

class SleepEventClassificationLayer(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(SleepEventClassificationLayer, self).__init__(**kwargs)
        self.bruxism_frame_classification_layer = BruxismClassificationLayer()
        self.snoring_frame_classification_layer = SnoringClassificationLayer()
        self.cough_frame_classification_layer = CoughClassificationLayer()
        self.speech_frame_judgement_layer = SpeechClassificationLayer()

    def call(self, scores):

        # scores= tf.squeeze(scores, axis=0)
        # print(f"scores 차원 : {scores.shape}")
        #42 ,521이 나와야함
        bruxism_result = self.bruxism_frame_classification_layer(scores)
        # tf.print("bruxism_result.shape : ",bruxism_result.shape)
        snoring_result = self.snoring_frame_classification_layer(scores)

        cough_result = self.cough_frame_classification_layer(scores)
        speech_result = self.speech_frame_judgement_layer(scores)

        # score만 뽑기 (float32 tensor로 캐스팅)
        bruxism_score = tf.cast(bruxism_result[1], tf.float32)
        snoring_score = tf.cast(snoring_result[1], tf.float32)
        cough_score = tf.cast(cough_result[1], tf.float32)
        speech_score = tf.cast(speech_result[1], tf.float32)

        # is_normal: 모든 event가 False이면 1.0, 하나라도 True면 0.0
        any_abnormal = tf.reduce_any([
            tf.cast(bruxism_result[0], tf.bool),
            tf.cast(snoring_result[0], tf.bool),
            tf.cast(cough_result[0], tf.bool),
            tf.cast(speech_result[0], tf.bool),
        ])
        is_normal = tf.cast(tf.logical_not(any_abnormal), tf.float32)
        outputs = tf.stack([bruxism_score, snoring_score, cough_score, speech_score, is_normal], axis=0)  # shape: (5,)
        outputs = tf.expand_dims(outputs, axis=0)  # shape: (1, 5)

        return outputs
