#!/bin/bash
key=/home/lht444/.ssh/id_rsa
user=tran.duy.vu@di.ku.dk
erdadir=' '
mnt=erda2/
if [ -f "$key" ]
then
    chmod +rwx $HOME
    mkdir -p ${mnt}
    sshfs ${user}@io.erda.dk:${erdadir} ${mnt} -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 -o IdentityFile=${key}
    chmod -rwx $HOME
    chmod u+rwx $HOME
else
    echo "'${key}' is not an ssh key"
fi
