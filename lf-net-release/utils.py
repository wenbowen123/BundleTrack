def embed_breakpoint(terminate=True):
    print('\nyou are inside a break point\n')
    embedding = ('import IPython\n'
                 'IPython.embed()\n'
                 )
    if terminate:
        embedding += (
            'assert 0, \'force termination\'\n'
        )

    return embedding

def print_opt(opt):
    content_list = []
    args = list(vars(opt))
    args.sort()
    for arg in args:
        content_list += [arg.rjust(25, ' ') + '  ' + str(getattr(opt, arg))]
    print_notification(content_list, 'OPTIONS')

def print_notification(content_list, notifi_type='NOTIFICATION'):
    print(('---------------------- {0} ----------------------'.format(notifi_type)))
    print()
    for content in content_list:
        print(content)
    print()
    print('----------------------------------------------------')
