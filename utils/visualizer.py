from visdom import Visdom
import json 

class Visualizer(object):
    """ Visualizer
    """
    def __init__(self, port='13579', env='main', id=None):
        self.cur_win = {}
        self.vis = Visdom(port=port, env=env)
        self.id = id
        self.env = env
        # Restore
        ori_win = self.vis.get_window_data()
        ori_win = json.loads(ori_win)
        #print(ori_win)
        self.cur_win = { v['title']: k for k, v in ori_win.items()  }

    def vis_scalar(self, name, x, y, opts=None):
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        
        if self.id is not None:
            name = "[%s]"%self.id + name
        default_opts = { 'title': name }
        if opts is not None:
            default_opts.update(opts)

        win = self.cur_win.get(name, None)
        if win is not None:
            self.vis.line( X=x, Y=y, opts=default_opts, update='append',win=win )
        else:
            self.cur_win[name] = self.vis.line( X=x, Y=y, opts=default_opts)

    def vis_image(self, name, img, env=None, opts=None):
        """ vis image in visdom
        """
        if env is None:
            env = self.env 
        if self.id is not None:
            name = "[%s]"%self.id + name
        win = self.cur_win.get(name, None)
        default_opts = { 'title': name }
        if opts is not None:
                default_opts.update(opts)
        if win is not None:
            self.vis.image( img=img, win=win, opts=opts, env=env )
        else:
            self.cur_win[name] = self.vis.image( img=img, opts=default_opts, env=env )

if __name__=='__main__':
    import numpy as np
    vis = Visualizer()
    vis.vis_scalar('test', [6], [7])
    vis.vis_image('test_image', np.zeros( (3,500,500) , dtype=np.float32) )
    