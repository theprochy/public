#include "ec.h"
#include "bits.h"
#include "ptab.h"
#include "memory.h"
#include "string.h"

#include <stdio.h>

typedef enum {
    sys_print      = 1,
    sys_sum        = 2,
    sys_break      = 3,
    sys_thr_create = 4,
    sys_thr_yeild  = 5,
} Syscall_numbers;

void Ec::syscall_handler (uint8 a)
{
    // Get access to registers stored during entering the system - see
    // entry_sysenter in entry.S
    Sys_regs * r = current->sys_regs();
    Syscall_numbers number = static_cast<Syscall_numbers> (a);

    switch (number) {
        case sys_print: {
            char *data = reinterpret_cast<char*>(r->esi);
            unsigned len = r->edi;
            for (unsigned i = 0; i < len; i++)
                printf("%c", data[i]);
            break;
        }
        case sys_break: {
            if (r->esi == 0) {
                r->eax = Ec::break_current;
                break;
            }
            if (r->esi > 0xC0000000 || r->esi < Ec::break_min) {
                r->eax = 0;
                break;
            }

            mword old_break = Ec::break_current;
            r->eax = Ec::break_current;
            if (r->esi > Ec::break_current) {
                if (Ec::break_current != (Ec::break_current & ~PAGE_MASK)) {
                    if (((r->esi - 1) >> PAGE_BITS) == (Ec::break_current >> PAGE_BITS)) {
                        memset(reinterpret_cast<void *> (Ec::break_current), 0, r->esi - Ec::break_current);
                        Ec::break_current = r->esi;
                        break;
                    }

                    memset(reinterpret_cast<void *> (Ec::break_current), 0,
                           (Ec::break_current & ~PAGE_MASK) + 0x1000 - Ec::break_current);
                }

                unsigned size = ((r->esi - 1 - Ec::break_current) >> PAGE_BITS) + 1;

                while (size > 0) {
                    if ((Ec::break_current & ~PAGE_MASK) != Ec::break_current) {
                        Ec::break_current &= ~PAGE_MASK;
                    } else {
                        void *ptr = Kalloc::allocator.alloc_page(1, Kalloc::FILL_0);
                        if (ptr == NULL) {
                            r->esi = old_break;
                            r->eax = 0;
                            break;
                        }

                        if (!Ptab::insert_mapping(Ec::break_current, Kalloc::virt2phys(ptr),
                                                Ptab::PRESENT | Ptab::RW | Ptab::USER)) {
                            r->esi = old_break;
                            r->eax = 0;
                            break;
                        }
                        size--;
                    }
                    Ec::break_current += PAGE_SIZE;
                }
            }

            if (r->esi < Ec::break_current) {
                if (((r->esi) >> PAGE_BITS) == ((Ec::break_current - 1) >> PAGE_BITS)) {
                    memset(reinterpret_cast<void *> (r->esi), 0, Ec::break_current - r->esi);
                    Ec::break_current = r->esi;
                    break;
                }

                mword virt = Ec::break_current & ~PAGE_MASK;
                if (virt == Ec::break_current) { virt -= 0x1000; }

                while (virt >= r->esi) {
                    mword mapped = Ptab::get_mapping(virt) & ~PAGE_MASK;
                    Kalloc::allocator.free_page(Kalloc::phys2virt(mapped));
                    if (!Ptab::insert_mapping (virt, 0x0, static_cast<mword> (0L))) {
                        r->eax = 0;
                        break;
                    }
                    virt -= PAGE_SIZE;
                }
            }
            Ec::break_current = r->esi;
            break;
        }
        default:
            printf ("unknown syscall %d\n", number);
            break;
    };

    ret_user_sysexit();
}
