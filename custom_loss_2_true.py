import tensorflow as tf
from keras import backend as K

class custom_loss:

    @tf.function
    def mase(self,y_true, y_pred):
        import math
        """
            Mean Absolute Scaled Error
            seasonality를 y_true에서 직접 data를 맞춰 넣어줘야된다.
            
            y_true 예시
            (batch_size,time_step,[기간평균오차를 계산할 대상, 예측에 대한 실제값])
            기간평균오차를 계산할 대상을 가지고, 실제 기간 gap에 대한 오차를 구하면 된다.

            -- 최종 공식 --
                비계절성 : 평균(절대값(y 현재 예측 - 현재 실제)) / 평균(절대값(실제 다음 값에 대한 오차))
                계절성 : 평균(절대값(y 현재 예측 - 현재 실제)) / 평균(절대값(실제 기간 gap에 대한 오차))
            평균에 절대값의 오차빼기는 mae와 같으므로, mae를 활용했다.
        """
        # def _naive_forecasting(actual, seasonality: int = 1):
        #     return actual[:-seasonality]

        def _error(actual, predicted):
            """ 실제값-예측값 """
            return actual - predicted

        def _mae(actual, predicted):
            return K.mean(K.abs(_error(actual, predicted)))

        denominator = _mae(y_true[:,:,0:1],y_true[:,:,1:2])

        if denominator == 0 or tf.math.is_nan(denominator):
            # 실제 값의 앞뒤가 똑같은경우, 분모가 0이되어버린다. (전후시간의 갭이 없음.)
            # 값이 홀수개가 들어와 미래의 값과 비교할 것이 없는 경우, 분모는 nan이 되버린다.
            # 이와같은경우, 분모를 MAE값을 0.111로 설정한다. (그냥 작은 값으로 설정함....)
            # 일단 비상용으로 소스를 이렇게 작성해놓긴 했으나, 실제 경험 상 해당 메세지가 뜨는 상황이 학습 간 1번이라도 발생하면 학습율이 현저하게 떨어진다.
            K.print_tensor('==================== someting STRANGE HAPPEN!!!!!!!!!!!!!!!!! ================')
            K.print_tensor(y_true,message='\ny_true==')
            K.print_tensor(y_pred,message='\ny_pred==')
            
            K.print_tensor(_mae(y_true[:,:,1:2], y_pred),message='\nChild==')
            K.print_tensor(denominator,message='\nParent==')
            
            return _mae(y_true, y_pred) / 1
        else:
            result = _mae(y_true, y_pred) / denominator
            K.print_tensor(result,message='\nIN_BATCH_LOSS==')
            return result