set nocompatible              " be iMproved, required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'
Plugin 'scrooloose/nerdtree'
Plugin 'w0rp/ale'
Plugin 'davidhalter/jedi-vim'
Plugin 'vim-airline/vim-airline'
Plugin 'adelarsq/vim-hackernews'

call vundle#end()            " required
filetype plugin indent on    " required
" To ignore plugin indent changes, instead use:
"filetype plugin on
"
" Brief help
" :PluginList       - lists configured plugins
" :PluginInstall    - installs plugins; append `!` to update or just :PluginUpdate
" :PluginSearch foo - searches for foo; append `!` to refresh local cache
" :PluginClean      - confirms removal of unused plugins; append `!` to auto-approve removal
"
" see :h vundle for more details or wiki for FAQ
" Put your non-Plugin stuff after this line
syntax enable 
set number
set ts=4
au FileType py set autoindent
au FileType py set smartindent
au FileType py set textwidth=79 " pep 8 stuff 
au FileType py set numberwidth=80
set showmatch
set cursorline
let python_highlight_all = 1
set shiftwidth=4
set expandtab
let g:airline#extension#ale#enabled = 1
set showcmd
set hlsearch
set incsearch
set wildmenu
set wildmode=full
nmap <F8> <Plug>(ale_fix)
noremap <F5> :w !python3 %<CR>
inoremap <F5> <ESC>:w !python3 %<CR>
let g:ale_python_flake8_executable = 'python3'
let g:ale_python_flake8_options = '-m flake8'
let g:ale_fixers = {'python': ['autopep8'],}
let g:completor_python_binary = '/usr/bin/python3.7'
"let g:completor_auto_close_doc = 0
let g:jedi#completions_enabled = 0
let g:jedi#use_tabs_not_buffers = 0
let g:jedi#use_splits_not_buffers = "top"
let g:jedi#show_call_signatures = "0"
nnoremap <C-J> <C-W><C-J>
nnoremap <C-K> <C-W><C-K>
nnoremap <C-L> <C-W><C-L>
nnoremap <C-H> <C-W><C-H>
