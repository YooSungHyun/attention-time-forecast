import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.activations import *
import math

''' 현재 어텐션의 한계
    한개의 어텐션 벨류밖에 국한된다.
    query, value의 전치와, softmax axis를 통하여, time-step과 feature들 중 선택으로 1가지 기준의 attention을 진행할 수 있다.
    다만 time-step으로 attention을 하던, feature로 attention을 하던 결국 reduce_sum에서 weigthed sum을 하는 과정을 거치기 때문에,
    결과적으로 time-step과 feature간의 attention 관계를 동시에 확인할 수 없다.(sum하면서 한쪽이 뭉개진다.)

    DARNN에서, encoding에서 feature를 보고, dec에서 time attention을 친 이유가 이 두개를 전부 보고 싶었기 때문이지 않을까 싶다...
'''

class attention_enc(Model):
    def __init__(self, units):
        super(attention_enc, self).__init__()
        self.lstm = LSTM(units,return_sequences=True, return_state=True)
        self.units = units

    def call(self, x):
        enc_output, enc_h, enc_c = self.lstm(x)
        return enc_output, enc_h, enc_c

class dot_product_dec(Model):
    def __init__(self,units):
        super(dot_product_dec, self).__init__()
        self.lstm = LSTM(units, return_sequences=True,return_state=True)
        # 어텐션을 사용합니다.
        self.attention = dot_product()
        # context vector의 concat이 진행되면, units 수 보다는 많은 input이 생성되게 되고, 그게 dense로 들어가면서 유실될 수 있는 부분들이
        # 우려스러워서 이와같이 dense의 units은 *2를 해서 적용시키게 된다.
        self.units = units*2
        self.dense1 = Dense(self.units, activation="tanh")

        # 기존의 문서에서는 tanh가 들어간 dense 1 layer 사용 이후 softmax activation을 사용했는데, 필자는 회귀니까 LeakyReLU로 깔아봤다.
        self.dense2 = Dense(math.ceil(self.units/2), activation=LeakyReLU())
        
        # batch_norm은 논문에선 사용되지 않았다, 혹시 몰라서 넣어볼까 싶어 활용함.
        self.batch_norm1 = BatchNormalization()
        self.batch_norm2 = BatchNormalization()
        
        # 교사강요를 사용하기위해 1 time-step끼리만 비교하게 된다.
        self.fc = Dense(1)

    def call(self, x,enc_output, enc_h, enc_c):
        # 인코더에서 출력된 값들을 통해 decoder 수행
        dec_output, dec_h, dec_c = self.lstm(x, initial_state=[enc_h, enc_c])

        # decoder에서 나온 값과, encoder의 state들을 attention
        context_vector, attention_weights = self.attention(dec_h, enc_output)
        decoder_combined_context = tf.concat([tf.expand_dims(context_vector, 1), dec_output],axis=-1)
        # (batch_size,lstm_output_feature) 와 (batch_size,1,lstm_output_feature)의 concat을 위해 context_vector의 차원을 1 늘려준다.
        # (batch_size,1,context_vector+decoder_lstm_feature) : 한 시점에 대한 attention 값이 모두 적용된, 학습 준비가 완료된 data set
        Wc = self.dense1(decoder_combined_context)
        Wy = self.dense2(Wc)
        output = self.fc(Wy)

        return output, attention_weights

class bahdanau_dec(Model):
    def __init__(self,units, lookup='v'):
        import math
        super(bahdanau_dec, self).__init__()
        self.lstm = LSTM(units, return_sequences=True,return_state=True)
        
        # 어텐션을 사용합니다.
        self.lookup = lookup
        self.attention = bahdanau(units, lookup)

        '''
        attention value야 뭘 기준으로 구하기만하면 그만이니까, 각각의 방법으로 돌려보았다.
        '''
        # horizontal 하게 보면, feature 기준
        if lookup == 'h':
            dense1_output = units
            dense2_output = math.ceil(units/2)
        # vertical 하게 보면, time_step 기준
        elif lookup == 'v':
            dense1_output = 64
            dense2_output = math.ceil(units/2)

        # 여기선 tanh도 않쓰고 성능 확실한 ReLU로 조짐
        self.dense1 = Dense(dense1_output, activation=LeakyReLU())
        self.dense2 = Dense(dense2_output, activation=LeakyReLU())

        self.batch_norm1 = BatchNormalization()
        self.batch_norm2 = BatchNormalization()
        self.fc = Dense(1)

    def call(self, x, enc_h, enc_output):
        # 바다나우는 t-1 디코더 상태로 attention 하고, 나온 context_vector를 현재 t의 concat하여 decoder를 돌린다.
        # 가장 처음 상태 t의 디코더는 t-1이 encoder의 은닉상태임을 생각하자.
        context_vector, attention_weights = self.attention(enc_h, enc_output)
        x = tf.concat([tf.expand_dims(context_vector, 1), x],axis=-1)

        # 위에서 결합된 벡터를 GRU에 전달합니다.
        dec_output, dec_h, dec_c = self.lstm(x)

        print("dec_output shape : ", dec_output.shape)
        dec_output = tf.reshape(dec_output, (-1, dec_output.shape[2]))
        print("change dec_output shape : ", dec_output.shape)
        hidden = self.batch_norm1(dec_output)
        hidden = self.dense1(hidden)
        hidden = self.batch_norm2(hidden)
        hidden = self.dense2(hidden)
        output = self.fc(hidden)
        print("output shape : ", output.shape)
        return output, [dec_h, dec_c], attention_weights

class dot_product(Layer):

    def __init__(self):
        super(dot_product,self).__init__()
        self.attention_value = None
        self.attention_weights = None

    def call(self, query, values):
        '''
        query : decoder hidden / batch_size, lstm_features
        values(==key) : encoder hidden / batch_size, max_length, lstm_features
        '''
        query = tf.expand_dims(query,axis=1) # 전치 효과내기
        score = dot([query,values],axes=[2,2])
        # 행렬곱 수행 시, (batch_size, score, time_step)가 되니까, 열변환을 수행하여, 기본적인 형태인 (batch_size, time_step, score)로 변환
        score = Permute((2, 1))(score)
        '''
        하단 주석 처리 부분은, BERT의 attention scaling 내용이다.
        내용이 뭔고하니, attention table의 dimension이 커질수록, softmax에서 score 를 먹일때 1에 수렴하게 먹이니, 다 고만고만해서 제대로 attention이 안되는걸,
        아쌀하게 스케일링 때려서 쓸놈만 쓰자는 주의이다.
        검토할 Attention table에 따라 유도리있게 조절해보는 것도 방법일듯 하다.
        '''
        # score = score * 1.0 / math.sqrt(float(values.shape[1]))
        
        self.attention_weights = tf.nn.softmax(score, axis=1)
        # batch_size, time_step, 1 : time_step에 대한 softmax score를 구할 수 있음. (axis=1은 시간열이 1열이기 때문.)
        
        self.attention_value = self.attention_weights * values
        # batch_size, time_step, lstm_output_features : 어텐션의 최종 결과값을 얻기 위해서 각 인코더의 은닉상태와 어텐션 가중치값들을 곱하고,
        
        self.attention_value = tf.reduce_sum(self.attention_value, axis=1)
        # batch_size, lstm_output_features : 가중치가 적용된 모든 시계열을 weighted sum하여, 특정 decoder 시계열 t에 대한 feature로 사용됨.
        return self.attention_value, self.attention_weights

class bahdanau(Layer):

    def __init__(self,units, lookup='v'):
        super(bahdanau,self).__init__()
        self.w1 = Dense(units)
        self.w2 = Dense(units)
        self.lookup = lookup
        # vertical하게 보게되면 시간들에 대한 1개의 score만 있으면 되고,
        if lookup=='v':
            dense_output=1
        # horizontal하게 보게되면 모든 시간에 대한 각 feature_len 만큼의 output이 필요하다.
        # 이 부분은 각 예측에 사용하고자 하는 feature_len에 따라 수정해주어야함.
        elif lookup=='h':
            dense_output=13
        self.v = Dense(dense_output)
        self.attention_value = None
        self.attention_weights = None

    def call(self, query, values):
        '''
        query : decoder hidden / batch_size, lstm_features
        values(==key) : encoder hidden / batch_size, max_length, lstm_features
        '''
        # vertical하게 보기 위해서는, softmax를 1열(time-step 기준)에 쳐야되고
        # reduce_sum역시 time-step으로 모두 더해 weigthed sum을 구해야한다.
        if self.lookup=='v':
            softmax_axis=1
            reduce_sum_axis = 1
        # horizontal하게 보기 위해서는, softmax를 2열(feature_len 기준)에 쳐야되고
        # reduce_sum역시 feature_len 기준으로 모두 더해 weigthed sum을 구해야한다.
        elif self.lookup=='h':
            softmax_axis=2
            reduce_sum_axis = 2

        query = tf.expand_dims(query,axis=1)
        # batch,1,lstm_features : 전치효과
        score = tf.nn.tanh(self.w1(values) + self.w2(query))
        # batch_size, time_step, lstm_output : 바다나우 논문에 대한 score 계산

        score = self.v(score)
        # batch,max_length,1 : 논문대로의 스코어 완성

        # dot_product에서 설명한 BERT Attention scailing
        # score = score * 1.0 / math.sqrt(float(values.shape[1]))

        # tf.nn.softmax의 axis default는 -1로 가장 맨 뒤 디멘션을 보게 된다 유의하자.
        self.attention_weights = tf.nn.softmax(score, axis=softmax_axis)
        # (v) batch_size, time_step, 1 : 시간별 계산된 weigths
        # (h) batch_size, 1, v_units_len : 특징별 계산된 weights (v의 units수로 그럴싸하게 맞춰야되는데 이게 학습이 잘 되는진 확인 안해봤다...;;)
        
        self.attention_value = self.attention_weights * values
        # (v) batch_size, time_step, lstm_features : time-step 기준의 attention_value
        # (h) batch_size, v_units_len, lstm_features : v_units_len(예상되기론 feature들) 기준의 attention_value

        # TODO : reduce_sum에서 뭉개지 않고 그대로 학습에 이어갈 수 있는 방법은 없을까?...그렇다면 1 attention으로도 잘 나올 것 같은데...
        self.attention_value = tf.reduce_sum(self.attention_value, axis=reduce_sum_axis)
        return self.attention_value, self.attention_weights

# 구현만 해놓음. 사용해보진 않음. dot_product를 대신하여 사용해볼 수 있을 것이라 사료됨.
class dot_product_general(Layer):
    def __init__(self,units):
        super(dot_product_general,self).__init__()
        self.w_complete = Dense(units)
        self.attention_value = None
        self.attention_weights = None

    def call(self, query, values):
        '''
        query : decoder hidden / batch_size, lstm_features
        values(==key) : encoder hidden / batch_size, max_length, lstm_features
        '''
        query = tf.expand_dims(query,axis=1) # 전치 효과내기
        score = dot([query,self.w_complete(values)],axes=[2,2])
        score = Permute((2, 1))(score)

        self.attention_weights = tf.nn.softmax(score, axis=1) # batch, max_len, 1 -> sequence에서 각 단어가 가지는 중요도!
        self.attention_value = self.attention_weights * values # batch, max_len, lstm_features
        self.attention_value = tf.reduce_sum(self.attention_value, axis=1)

        return self.attention_value, self.attention_weights
