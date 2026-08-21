#ifndef MODEL_HPP
#define MODEL_HPP

#include "iten.hpp"


class BigModel : public Module {
		public:
				Linear* linear1;
				ReLU* relu1;
				Linear* linear2;
				ReLU* relu2;
				Linear* linear3;
				Softmax* softmax;

				BigModel();
				~BigModel();
				Tensor* forward(Tensor* input) override;
};



#endif
