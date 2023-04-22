
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        from .cycle_gan_model import CycleGANModel
        #assert(opt.align_data == False)
        model = CycleGANModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
